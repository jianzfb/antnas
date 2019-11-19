# -*- coding: UTF-8 -*-
# @Time    : 2019-09-14 11:25
# @File    : SegSN.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from nas.interfaces.NetworkBlock import *
from nas.interfaces.NetworkCell import *
from nas.networks.StochasticSuperNetwork import StochasticSuperNetwork
from nas.networks.EvolutionSuperNetwork import EvolutionSuperNetwork
import networkx as nx
import numpy as np
import torch.nn.functional as F
from torch import nn
import torch
from torchvision import transforms
from nas.utils.drawers.BSNDrawer import BSNDrawer
from nas.implem.Loss import *
from nas.implem.SegmentationAccuracyEvaluator import *
from nas.interfaces.AdvancedNetworkBlock import *

__all__ = ['BiSegSN']


class SegOutLayer(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, in_chan, out_shape):
        super(SegOutLayer, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_shape[0], kernel_size=1, stride=1, padding=0, bias=True)
        self.out_shape = out_shape
        self.params = {
            'module_list': ['SegOutLayer'],
            'name_list': ['SegOutLayer'],
            'SegOutLayer': {'out_shape': out_shape},
            'out': 'outname'
        }

    def forward(self, input):
        y = self.conv(input)
        y = F.upsample(y, scale_factor=4.0, mode='bilinear')
        return y

    def get_flop_cost(self, x):
        return [0] + [0] * (self.state_num - 1)


class BiSegSN(EvolutionSuperNetwork):
    _INPUT_NODE_FORMAT = 'I_{}_{}'              # 不可学习
    _OUTPUT_NODE_FORMAT = 'O_{}_{}'             # 不可学习
    _AGGREGATION_NODE_FORMAT = 'A_{}_{}'        # 不可学习
    _CELL_NODE_FORMAT = 'CELL_{}_{}'            # 可学习  (多种状态)
    _TRANSFORMATION_FORMAT = 'T_{}_{}-{}_{}'    # 可学习 （激活/不激活）
    _LINK_FORMAT = 'L_{}_{}-{}_{}'              # 不可学习
    _FIXED_NODE_FORMAT = 'F_{}_{}'              # 不可学习

    def __init__(self,
                 blocks_per_stage,
                 cells_per_block,
                 channels_per_block,
                 data_prop,
                 static_proba, *args, **kwargs):
        super(BiSegSN, self).__init__(*args, **kwargs)
        NetworkBlock.state_num = 5
        self.in_chan = data_prop['in_channels']
        self.in_size = data_prop['img_dim']
        self.out_dim = data_prop['out_size'][0]

        self.static_node_proba = static_proba
        self._input_size = (self.in_chan, self.in_size, self.in_size)

        self.blocks = nn.ModuleList([])
        self.graph = nx.DiGraph()
        self.sampling_parameters = nn.ParameterList()
        self._loss = cross_entropy
        self._accuracy_evaluator = SegmentationAccuracyEvaluator(class_num=self.out_dim)

        # 1.step encoder
        # head (固定计算节点，对应激活参数不可学习)
        in_module = ConvBn(self.in_chan, 16, k_size=3, stride=2, relu=True)
        in_name = self.add_aggregation((0, 0), module=in_module, node_format=self._INPUT_NODE_FORMAT)

        # search space（stage - block - cell）
        # backbone
        pos_offset = 1
        offset_per_stage = []
        for stage_i in range(len(blocks_per_stage)):
            offset_per_stage.append(pos_offset)
            pos_offset = self.add_stage(pos_offset,
                                        blocks_per_stage[stage_i],
                                        cells_per_block[stage_i],
                                        channels_per_block[stage_i],
                                        16 if stage_i == 0 else channels_per_block[stage_i - 1][-1],
                                        stage_i == len(blocks_per_stage) - 1)

            if stage_i > 0:
                # add stage edge
                self.graph.add_edge(self._CELL_NODE_FORMAT.format(*(0, offset_per_stage[stage_i-1]+sum(cells_per_block[stage_i-1])*2-1)),
                                    self._AGGREGATION_NODE_FORMAT.format(*(0, offset_per_stage[stage_i])),
                                    width_node=self._AGGREGATION_NODE_FORMAT.format(*(0, offset_per_stage[stage_i])))

        # 1.1.step link last cell to ASPP
        last_stage_index = len(blocks_per_stage) - 1
        encoder_last_layer = (0, offset_per_stage[last_stage_index] + sum(cells_per_block[last_stage_index]) * 2 - 1)
        aspp_node_pos = (1, last_stage_index)
        self.add_aggregation(aspp_node_pos,
                             ASPPBlock(channels_per_block[last_stage_index][-1], 256, atrous_rates=[3, 6, 9]),
                             node_format=self._FIXED_NODE_FORMAT)

        self.graph.add_edge(self._CELL_NODE_FORMAT.format(*encoder_last_layer),
                            self._FIXED_NODE_FORMAT.format(*aspp_node_pos),
                            width_node=self._FIXED_NODE_FORMAT.format(*aspp_node_pos))

        # 1.2.step link head to search space
        self.graph.add_edge(self._INPUT_NODE_FORMAT.format(0, 0),
                            self._AGGREGATION_NODE_FORMAT.format(0, 1),
                            width_node=self._AGGREGATION_NODE_FORMAT.format(0, 1))

        # 2.step decoder
        # 2.1.step aspp decoder
        aspp_decoder_pos = (2, last_stage_index)
        self.add_aggregation(aspp_decoder_pos,
                             ResizedBlock(256, -1, scale_factor=4),
                             node_format=self._FIXED_NODE_FORMAT)
        self.graph.add_edge(self._FIXED_NODE_FORMAT.format(*aspp_node_pos),
                            self._FIXED_NODE_FORMAT.format(*aspp_decoder_pos),
                            width_node=self._FIXED_NODE_FORMAT.format(*aspp_decoder_pos))

        # 2.2.step middle feature decoder
        middle_branch_pos = (0, offset_per_stage[last_stage_index-3] + sum(cells_per_block[last_stage_index-3]) * 2 - 1)

        middle_branch_decoder_pos = (1, last_stage_index-3)
        self.add_aggregation(middle_branch_decoder_pos,
                             ConvBn(channels_per_block[last_stage_index-3][-1], 48, relu=True, k_size=3),
                             node_format=self._FIXED_NODE_FORMAT)
        self.graph.add_edge(self._CELL_NODE_FORMAT.format(*middle_branch_pos),
                            self._FIXED_NODE_FORMAT.format(*middle_branch_decoder_pos),
                            width_node=self._FIXED_NODE_FORMAT.format(*middle_branch_decoder_pos))

        decoder_aggregation_node_pos = (2, last_stage_index-3)
        self.add_aggregation(decoder_aggregation_node_pos,
                             ConcatBlock(),
                             node_format=self._AGGREGATION_NODE_FORMAT)

        self.graph.add_edge(self._FIXED_NODE_FORMAT.format(*middle_branch_decoder_pos),
                            self._AGGREGATION_NODE_FORMAT.format(*decoder_aggregation_node_pos),
                            width_node=self._AGGREGATION_NODE_FORMAT.format(*decoder_aggregation_node_pos))

        self.graph.add_edge(self._FIXED_NODE_FORMAT.format(*aspp_decoder_pos),
                            self._AGGREGATION_NODE_FORMAT.format(*decoder_aggregation_node_pos),
                            width_node=self._AGGREGATION_NODE_FORMAT.format(*decoder_aggregation_node_pos))

        decoder_aggregation_conv_1_node_pos = (3, last_stage_index-3)
        self.add_aggregation(decoder_aggregation_conv_1_node_pos,
                             SepConvBN(256+48, 256, relu=True, k_size=3),
                             node_format=self._FIXED_NODE_FORMAT)

        self.graph.add_edge(self._AGGREGATION_NODE_FORMAT.format(*decoder_aggregation_node_pos),
                            self._FIXED_NODE_FORMAT.format(*decoder_aggregation_conv_1_node_pos),
                            width_node=self._FIXED_NODE_FORMAT.format(*decoder_aggregation_conv_1_node_pos))

        decoder_aggregation_conv_2_node_pos = (4, last_stage_index-3)
        self.add_aggregation(decoder_aggregation_conv_2_node_pos,
                             SepConvBN(256, 256, relu=True, k_size=3),
                             node_format=self._FIXED_NODE_FORMAT)

        self.graph.add_edge(self._FIXED_NODE_FORMAT.format(*decoder_aggregation_conv_1_node_pos),
                            self._FIXED_NODE_FORMAT.format(*decoder_aggregation_conv_2_node_pos),
                            width_node=self._FIXED_NODE_FORMAT.format(*decoder_aggregation_conv_2_node_pos))

        # 3.step output(固定计算节点，对应激活参数不可学习)
        out_module = SegOutLayer(256, data_prop['out_size'])
        out_name = self._OUTPUT_NODE_FORMAT.format(*(0, offset_per_stage[-1]+sum(cells_per_block[-1])*2))
        sampling_param = self.sampling_param_generator(out_name)

        self.graph.add_node(out_name,
                            module=len(self.blocks),
                            module_params=out_module.params,
                            sampling_param=len(self.sampling_parameters),
                            pos=BSNDrawer.get_draw_pos(pos=(0, offset_per_stage[-1]+sum(cells_per_block[-1])*2)))
        self.graph.add_edge(self._FIXED_NODE_FORMAT.format(*decoder_aggregation_conv_2_node_pos),
                            out_name,
                            width_node=out_name)
        self.sampling_parameters.append(sampling_param)
        self.blocks.append(out_module)

        # set graph
        self.set_graph(self.graph, in_name, out_name)

    def add_stage(self, pos_offset, block_num, cells_per_block, channles_per_block, pre_stage_channels, is_final_stage):
        stage_offset = pos_offset
        offset_per_block = []
        for block_i in range(block_num):
            offset_per_block.append(stage_offset)
            if block_i == 0:
                self.add_block(stage_offset,
                               cells_per_block[block_i],
                               channles_per_block[block_i],
                               pre_stage_channels,
                               is_final_stage,
                               True if block_i == 0 else False)
            else:
                self.add_block(stage_offset,
                               cells_per_block[block_i],
                               channles_per_block[block_i],
                               channles_per_block[block_i-1],
                               is_final_stage,
                               False)

            stage_offset += cells_per_block[block_i] * 2

            # dense connection among blocks
            if block_i > 0:
                for pre_block_i in range(block_i):
                    if pre_block_i == block_i - 1:
                        # 固定连接
                        self.graph.add_edge(self._CELL_NODE_FORMAT.format(0, offset_per_block[pre_block_i] + cells_per_block[pre_block_i] * 2 - 1),
                                            self._AGGREGATION_NODE_FORMAT.format(0, offset_per_block[block_i] + 0 * 2),
                                            width_node=self._AGGREGATION_NODE_FORMAT.format(0, offset_per_block[block_i] + 0 * 2))

        return stage_offset

    def add_block(self, pos_offset, cells, channles, pre_block_channels, is_final_stage, reduction=False):
        for cell_i in range(cells):
            # Add
            self.add_aggregation((0, pos_offset+cell_i*2), AddBlock(), self._AGGREGATION_NODE_FORMAT)

            # Cell
            if not is_final_stage:
                self.add_cell((0, pos_offset + cell_i * 2 + 1),
                              CellBlock(pre_block_channels if cell_i == 0 else channles,
                                        channles,
                                        reduction=reduction if cell_i == 0 else False),
                              self._CELL_NODE_FORMAT)
            else:
                self.add_cell((0, pos_offset + cell_i * 2 + 1),
                              DilationCellBlock(pre_block_channels if cell_i == 0 else channles,
                                        channles,
                                        reduction=False),
                              self._CELL_NODE_FORMAT)

            # 固定连接
            self.graph.add_edge(self._AGGREGATION_NODE_FORMAT.format(0, pos_offset+cell_i*2),
                                self._CELL_NODE_FORMAT.format(0, pos_offset+cell_i*2+1),
                                width_node=self._CELL_NODE_FORMAT.format(0, pos_offset+cell_i*2+1))

            # dense connection among cells
            if cell_i > 0:
                for pre_cell_i in range(cell_i):
                    if pre_cell_i == cell_i - 1:
                        # 固定连接
                        self.graph.add_edge(self._CELL_NODE_FORMAT.format(0, pos_offset+pre_cell_i*2+1),
                                            self._AGGREGATION_NODE_FORMAT.format(0, pos_offset+cell_i*2),
                                            width_node=self._AGGREGATION_NODE_FORMAT.format(0, pos_offset+cell_i*2))

    def add_aggregation(self, pos, module, node_format):
        agg_node_name = node_format.format(*pos)
        sampling_param = self.sampling_param_generator(agg_node_name)

        self.graph.add_node(agg_node_name,
                            module=len(self.blocks),
                            module_params=module.params,
                            sampling_param=len(self.sampling_parameters),
                            pos=BSNDrawer.get_draw_pos(pos=pos))

        if sampling_param is not None:
            self.sampling_parameters.append(sampling_param)
        self.blocks.append(module)
        return agg_node_name

    def add_cell(self, pos, module, node_format):
        cell_node_name = node_format.format(*pos)
        sampling_param = self.sampling_param_generator(cell_node_name)
        self.graph.add_node(cell_node_name,
                            module=len(self.blocks),
                            module_params=module.params,
                            sampling_param=len(self.sampling_parameters),
                            pos=BSNDrawer.get_draw_pos(pos=pos))

        self.sampling_parameters.append(sampling_param)
        self.blocks.append(module)
        return cell_node_name

    def add_transformation(self, source, dest, module, src_node_format, des_node_format, transform_format, pos_shift=0):
        src_l, src_s = source
        dst_l, dst_s = dest

        trans_name = transform_format.format(src_l, src_s, dst_l, dst_s)
        source_name = src_node_format.format(src_l, src_s)
        dest_name = des_node_format.format(dst_l, dst_s)

        pos = BSNDrawer.get_draw_pos(source=source, dest=dest, pos_shift=pos_shift)
        sampling_param = self.sampling_param_generator(trans_name)

        self.graph.add_node(trans_name, module=len(self.blocks), module_params=module.params,
                            sampling_param=len(self.sampling_parameters), pos=pos)
        self.graph.add_edge(source_name, trans_name,  width_node=trans_name)
        self.graph.add_edge(trans_name, dest_name,  width_node=trans_name)
        self.sampling_parameters.append(sampling_param)
        self.blocks.append(module)
        return trans_name

    def loss(self, predictions, labels):
        return self._loss(predictions, labels)

    def accuray(self, predictions, labels):
        return self._accuracy_evaluator.accuracy(predictions, labels)
