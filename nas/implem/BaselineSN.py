# -*- coding: UTF-8 -*-
# @Time    : 2019-07-24 11:39
# @File    : BaselineSN.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from numbers import Number
import networkx as nx
import numpy as np
import torch.nn.functional as F
from torch import nn
import torch
from nas.interfaces.NetworkBlock import *
from nas.interfaces.NetworkCell import *
# from nas.networks.StochasticSuperNetwork import StochasticSuperNetwork
from nas.networks.EvolutionSuperNetwork import EvolutionSuperNetwork
from nas.utils.drawers.BSNDrawer import BSNDrawer
from nas.implem.Loss import *
from nas.implem.ClassificationAccuracyEvaluator import *

__all__ = ['BaselineSN']


class OutLayer(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, in_chan, out_shape, bias=True):
        super(OutLayer, self).__init__()
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.conv_1 = nn.Conv2d(in_chan, 960, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn = nn.BatchNorm2d(960)

        self.conv_2 = nn.Conv2d(960, 1280, kernel_size=1, stride=1, padding=0, bias=bias)
        self.conv_3 = nn.Conv2d(1280, out_shape[0], kernel_size=1, stride=1, padding=0, bias=bias)
        self.out_shape = out_shape
        self.params = {
            'module_list': ['OutLayer'],
            'name_list': ['OutLayer'],
            'OutLayer': {'out_shape': out_shape},
            'out': 'outname'
        }

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn(x)
        x = F.relu6(x)

        x = self.global_pool(x)
        x = self.conv_2(x)
        x = F.relu6(x)

        x = self.conv_3(x)
        return x.view(-1, *self.out_shape)

    def get_flop_cost(self, x):
        return [0] + [0] * (self.state_num - 1)


class BaselineSN(EvolutionSuperNetwork):
    _INPUT_NODE_FORMAT = 'I_{}_{}'              # 不可学习
    _OUTPUT_NODE_FORMAT = 'O_{}_{}'             # 不可学习
    _AGGREGATION_NODE_FORMAT = 'A_{}_{}'        # 不可学习
    _CELL_NODE_FORMAT = 'CELL_{}_{}'            # 可学习  (多种状态)
    _TRANSFORMATION_FORMAT = 'T_{}_{}-{}_{}'    # 可学习 （激活/不激活）
    _LINK_FORMAT = 'L_{}_{}-{}_{}'              # 不可学习

    def __init__(self,
                 blocks_per_stage,
                 cells_per_block,
                 channels_per_block,
                 data_prop,
                 static_proba, *args, **kwargs):
        super(BaselineSN, self).__init__(*args, **kwargs)
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
        self._accuracy_evaluator = ClassificationAccuracyEvaluator()

        # head (固定计算节点，对应激活参数不可学习)
        in_module = ConvBn(self.in_chan, channels_per_block[0][0], k_size=3, stride=2, relu=True)
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
                                        channels_per_block[stage_i+1][0] if stage_i != len(blocks_per_stage) - 1 else None,
                                        stage_i)

            if stage_i > 0:
                # add stage edge
                self.graph.add_edge(self._CELL_NODE_FORMAT.format(*(0, offset_per_stage[stage_i-1]+sum(cells_per_block[stage_i-1])*2-1)),
                                    self._AGGREGATION_NODE_FORMAT.format(*(0, offset_per_stage[stage_i])),
                                    width_node=self._AGGREGATION_NODE_FORMAT.format(*(0, offset_per_stage[stage_i])))

        # link head to search space
        self.graph.add_edge(self._INPUT_NODE_FORMAT.format(0, 0),
                            self._AGGREGATION_NODE_FORMAT.format(0, 1),
                            width_node=self._AGGREGATION_NODE_FORMAT.format(0, 1))

        # tail (固定计算节点，对应激活参数不可学习)
        # output layer
        out_module = OutLayer(channels_per_block[-1][-1], data_prop['out_size'], True)
        out_name = self._OUTPUT_NODE_FORMAT.format(*(0, offset_per_stage[-1]+sum(cells_per_block[-1])*2))
        sampling_param = self.sampling_param_generator(out_name)
        self.graph.add_node(out_name,
                            module=len(self.blocks),
                            module_params=out_module.params,
                            sampling_param=len(self.sampling_parameters),
                            pos=BSNDrawer.get_draw_pos(pos=(0, offset_per_stage[-1]+sum(cells_per_block[-1])*2)))
        self.graph.add_edge(self._CELL_NODE_FORMAT.format(*(0, offset_per_stage[-1] + sum(cells_per_block[-1]) * 2 - 1)),
                            out_name,
                            width_node=out_name)
        self.sampling_parameters.append(sampling_param)
        self.blocks.append(out_module)

        # set graph
        self.set_graph(self.graph, in_name, out_name)

    def add_stage(self, pos_offset, block_num, cells_per_block, channles_per_block, next_stage_channels, stage_index):
        stage_offset = pos_offset
        offset_per_block = []
        for block_i in range(block_num):
            offset_per_block.append(stage_offset)
            if block_i < block_num - 1:
                self.add_block(stage_offset,
                               cells_per_block[block_i],
                               channles_per_block[block_i],
                               channles_per_block[block_i + 1],
                               True if block_i == 0 and stage_index != 0 else False)
            else:
                self.add_block(stage_offset,
                               cells_per_block[block_i],
                               channles_per_block[block_i],
                               next_stage_channels,
                               True if block_i == 0 and stage_index != 0 else False)
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

    def add_block(self, pos_offset, cells, channles, next_block_channels, reduction=False):
        for cell_i in range(cells):
            # Add
            self.add_aggregation((0, pos_offset+cell_i*2), AddBlock(), self._AGGREGATION_NODE_FORMAT)

            # Cell
            if cell_i != cells - 1:
                self.add_cell((0, pos_offset+cell_i*2+1),
                              CellBlock(channles, channles, reduction=reduction if cell_i == 0 else False),
                              self._CELL_NODE_FORMAT)
            else:
                if next_block_channels is None:
                    next_block_channels = channles
                self.add_cell((0, pos_offset + cell_i * 2 + 1),
                              CellBlock(channles, next_block_channels, reduction=reduction if cell_i == 0 else False),
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


