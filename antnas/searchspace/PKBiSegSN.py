# -*- coding: UTF-8 -*-
# @Time    : 2020/7/20 6:08 下午
# @File    : PKBiSegSN.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import networkx as nx
from antnas.component.NetworkCell import *
from antnas.networks.UniformSamplingSuperNetwork import UniformSamplingSuperNetwork
from antnas.utils.drawers.NASDrawer import NASDrawer
from antnas.component.Loss import *
from antnas.component.ClassificationAccuracyEvaluator import *
from antnas.component.SegmentationAccuracyEvaluator import *
from antnas.searchspace.PKMixArc import *
__all__ = ['PKBiSegSN']


class BiSegDecoderCellBlock(NetworkBlock):
    n_layers = 3
    n_comp_steps = 1

    def __init__(self, channles, out_channels, reduction=False):
        super(BiSegDecoderCellBlock, self).__init__()
        self.channles = channles
        self.op_list = nn.ModuleList()
        self.params = {
            'module_list': ['Skip',
                            'ConvBn',
                            'ConvBn',
                            'SepConvBN',
                            'SepConvBN'],
            'name_list': ['Skip',
                          'ConvBn_k1',
                          'ConvBn_k3',
                          'SepConvBN_k3',
                          'SepConvBN_k5'],
            'Skip': {'in_chan': channles,
                     'out_chan': out_channels,
                     'reduction': False},
            'ConvBn_k1': {'in_chan': channles,
                          'k_size': 1,
                          'out_chan': out_channels,
                          'relu': True,
                          'stride': 1},
            'ConvBn_k3': {'in_chan': channles,
                          'k_size': 3,
                          'out_chan': out_channels,
                          'relu': True,
                          'stride': 1},
            'SepConvBN_k3': {'in_chan': channles,
                             'k_size': 3,
                             'out_chan': out_channels,
                             'relu': True,
                             'stride': 1},
            'SepConvBN_k5': {'in_chan': channles,
                             'k_size': 5,
                             'out_chan': out_channels,
                             'relu': True,
                             'stride': 1},
            'in_chan': channles,
            'out_chan': out_channels
        }
        self.structure_fixed = True

        self.op_list = self.build()
        assert(len(self.op_list) == NetworkBlock.state_num)

    def forward(self, input, sampling=None):
        if sampling is None:
            # only for ini
            return self.op_list[0](input)

        op_index = (int)(sampling.item())
        result = self.op_list[op_index](input)
        return result

    def get_flop_cost(self, x):
        cost_list = [0]
        for i in range(len(self.op_list) - 1):
            cost_list.append(self.op_list[i + 1].get_flop_cost(x)[1])

        return cost_list

    def get_latency(self, x):
        cost_list = [0]
        for i in range(len(self.op_list) - 1):
            cost_list.append(self.op_list[i + 1].get_latency(x)[1])

        return cost_list

    def get_param_num(self, x):
        cost_list = [0]
        for i in range(len(self.op_list) - 1):
            cost_list.append(self.op_list[i + 1].get_param_num(x)[1])

        return cost_list


class BiSegASPPCell(ASPPBlock):
    n_layers = 3
    n_comp_steps = 1

    def __init__(self, in_chan, depth):
        super(BiSegASPPCell, self).__init__(in_chan, depth, [2, 4, 6])


class BiSegDetailCellBlock(NetworkBlock):
    n_layers = 3
    n_comp_steps = 1

    def __init__(self, channles, out_channels, reduction=False):
        super(BiSegDetailCellBlock, self).__init__()
        self.channles = channles
        self.op_list = nn.ModuleList()
        self.params = {
            'module_list': ['ConvBn',
                            'SepConvBN',
                            'SepConvBN',
                            'SepConvBN',
                            'SepConvBN'],
            'name_list': ['ConvBn_k3',
                          'SepConvBN_k3',
                          'SepConvBN_k3_d2',
                          'SepConvBN_k3_d4',
                          'SepConvBN_k3_d6'],
            'ConvBn_k3': {'in_chan': channles,
                          'k_size': 3,
                          'out_chan': out_channels,
                          'relu': True,
                          'stride': 2 if reduction else 1},
            'SepConvBN_k3': {'in_chan': channles,
                             'k_size': 3,
                             'out_chan': out_channels,
                             'relu': True,
                             'stride': 2 if reduction else 1},
            'SepConvBN_k3_d2': {'in_chan': channles,
                             'k_size': 3,
                             'out_chan': out_channels,
                             'relu': True,
                             'stride': 2 if reduction else 1,
                             'dilation': 2},
            'SepConvBN_k3_d4': {'in_chan': channles,
                                'k_size': 3,
                                'out_chan': out_channels,
                                'relu': True,
                                'stride': 2 if reduction else 1,
                                'dilation': 4},
            'SepConvBN_k3_d6': {'in_chan': channles,
                                'k_size': 3,
                                'out_chan': out_channels,
                                'relu': True,
                                'stride': 2 if reduction else 1,
                                'dilation': 6},
            'in_chan': channles,
            'out_chan': out_channels
        }
        self.structure_fixed = True

        self.op_list = self.build()
        assert(len(self.op_list) == NetworkBlock.state_num)

    def forward(self, input, sampling=None):
        if sampling is None:
            # only for ini
            return self.op_list[0](input)

        op_index = (int)(sampling.item())
        result = self.op_list[op_index](input)
        return result

    def get_flop_cost(self, x):
        cost_list = [0]
        for i in range(len(self.op_list) - 1):
            cost_list.append(self.op_list[i + 1].get_flop_cost(x)[1])

        return cost_list

    def get_latency(self, x):
        cost_list = [0]
        for i in range(len(self.op_list) - 1):
            cost_list.append(self.op_list[i + 1].get_latency(x)[1])

        return cost_list

    def get_param_num(self, x):
        cost_list = [0]
        for i in range(len(self.op_list) - 1):
            cost_list.append(self.op_list[i + 1].get_param_num(x)[1])

        return cost_list


class PKBiSegSN(UniformSamplingSuperNetwork):
    def __init__(self, data_prop, *args, **kwargs):
        super(PKBiSegSN, self).__init__(*args, **kwargs)
        self.in_chan = data_prop['in_channels']
        self.in_size = data_prop['img_dim']
        self.out_dim = data_prop['out_size'][0]

        self.graph = nx.DiGraph()
        self.sampling_parameters = nn.ParameterList()
        self.blocks = nn.ModuleList([])

        self._loss = cross_entropy
        self._accuracy_evaluator = SegmentationAccuracyEvaluator(class_num=self.out_dim)

        # 输入节点
        identity = Identity(self.in_chan, self.in_chan)
        identity_name = SuperNetwork._INPUT_NODE_FORMAT.format(0, -2)
        self.graph.add_node(identity_name,
                            module=len(self.blocks),
                            module_params=identity.params,
                            sampling_param=len(self.blocks),
                            structure_fixed=identity.structure_fixed,
                            pos=NASDrawer.get_draw_pos(pos=(0, -2)),
                            sampled=1)
        self.blocks.append(identity)

        # 固定节点
        pool = AvgPoolingBlock(k_size=2, stride=2)
        pool_name = SuperNetwork._FIXED_NODE_FORMAT.format(0, -1)
        self.graph.add_node(pool_name,
                            module=len(self.blocks),
                            module_params=pool.params,
                            sampling_param=len(self.blocks),
                            structure_fixed=pool.structure_fixed,
                            pos=NASDrawer.get_draw_pos(pos=(0, -1)),
                            sampled=1)
        self.blocks.append(pool)

        self.graph.add_edge(identity_name,
                            pool_name,
                            width_node=pool_name)

        # backbone head
        head = ConvBn(self.in_chan, 32, k_size=3, stride=2, relu=True)
        # tail (固定计算节点，结构不可学习)
        tail = kwargs['out_layer']

        # 双分支
        # 分支1：主干结构设置为mobilenet-v2-0.5 + aspp
        # 分支2：detail
        self.seg_arc = PKMixArc(BiSegDecoderCellBlock,
                                BiSegASPPCell,
                                AddBlock,
                                ConvBn,
                                self.graph,
                                self.blocks,
                                decoder_input_endpoints=[2, 16],
                                decoder_input_strides=[4, 16],
                                decoder_depth=64,
                                backbone='mobilenetv2-0.5')
        in_name, out_name = self.seg_arc.generate(head, tail, self.detail_branch)

        self.graph.add_edge(pool_name,
                            in_name,
                            width_node=in_name)

        # set graph
        self.set_graph(self.graph, identity_name, out_name)

    def detail_branch(self, endpoint, pos_offset):
        detail_input = SuperNetwork._INPUT_NODE_FORMAT.format(0, -2)

        # stage 1 (384 x 384->192 x 192)
        for index in range(3):
            if index == 0:
                self.seg_arc.add_cell((0, pos_offset),
                              BiSegDetailCellBlock(3,
                                                   32,
                                                   reduction=True),
                              SuperNetwork._CELL_NODE_FORMAT)

                self.graph.add_edge(detail_input,
                                    SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset),
                                    width_node=SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset))

                pos_offset += 1
            else:
                self.seg_arc.add_cell((0, pos_offset),
                              BiSegDetailCellBlock(32,
                                                   32,
                                                   reduction=False),
                              SuperNetwork._CELL_NODE_FORMAT)

                self.graph.add_edge(SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset-1),
                                    SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset),
                                    width_node=SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset))

                pos_offset += 1

        # stage 2 (192 x 192 -> 96 x 96)
        for index in range(3):
            if index == 0:
                self.seg_arc.add_cell((0, pos_offset),
                              BiSegDetailCellBlock(32,
                                                   64,
                                                   reduction=True),
                              SuperNetwork._CELL_NODE_FORMAT)

                self.graph.add_edge(SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset-1),
                                    SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset),
                                    width_node=SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset))

                pos_offset += 1
            else:
                self.seg_arc.add_cell((0, pos_offset),
                              BiSegDetailCellBlock(64,
                                                   64,
                                                   reduction=False),
                              SuperNetwork._CELL_NODE_FORMAT)

                self.graph.add_edge(SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset - 1),
                                    SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset),
                                    width_node=SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset))

                pos_offset += 1

        # stage 3 (96 x 96 -> 48 x 48)
        for index in range(3):
            if index == 0:
                self.seg_arc.add_cell((0, pos_offset),
                              BiSegDetailCellBlock(64,
                                                   128,
                                                   reduction=True),
                              SuperNetwork._CELL_NODE_FORMAT)

                self.graph.add_edge(SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset-1),
                                    SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset),
                                    width_node=SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset))

                pos_offset += 1
            else:
                self.seg_arc.add_cell((0, pos_offset),
                              BiSegDetailCellBlock(128,
                                                   128,
                                                   reduction=False),
                              SuperNetwork._CELL_NODE_FORMAT)

                self.graph.add_edge(SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset-1),
                                    SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset),
                                    width_node=SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset))

                pos_offset += 1

        # 将endpoint 和 detail output合并
        self.seg_arc.add_fixed((0, pos_offset), ConcatBlock())
        self.graph.add_edge(endpoint,
                            SuperNetwork._FIXED_NODE_FORMAT.format(0, pos_offset),
                            width_node=SuperNetwork._FIXED_NODE_FORMAT.format(0, pos_offset))
        self.graph.add_edge(SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset-1),
                            SuperNetwork._FIXED_NODE_FORMAT.format(0, pos_offset),
                            width_node=SuperNetwork._FIXED_NODE_FORMAT.format(0, pos_offset))

        pos_offset += 1
        return SuperNetwork._FIXED_NODE_FORMAT.format(0, pos_offset-1), pos_offset

    def loss(self, predictions, labels):
        return self._loss(predictions, labels)

    def accuray(self, predictions, labels):
        return self._accuracy_evaluator.accuracy(predictions, labels)

    def hierarchical(self):
        return self.seg_arc.hierarchical