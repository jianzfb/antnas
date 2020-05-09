# -*- coding: UTF-8 -*-
# @Time    : 2020-04-21 09:15
# @File    : PKCifar10SN.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import networkx as nx
from nas.component.NetworkCell import *
from nas.networks.UniformSamplingSuperNetwork import UniformSamplingSuperNetwork
from nas.networks.AnchorsUniformSamplingSuperNetwork import *
from nas.networks.EvolutionSuperNetwork import *
from nas.utils.drawers.NASDrawer import NASDrawer
from nas.component.Loss import *
from nas.component.ClassificationAccuracyEvaluator import *
from nas.searchspace.StageBlockCellArc import *

__all__ = ['PKCifar10SN']


class Cifar10CellBlock(NetworkBlock):
    n_layers = 3
    n_comp_steps = 1

    def __init__(self, channles, out_channels, reduction=False):
        super(Cifar10CellBlock, self).__init__()
        self.channles = channles
        self.op_list = nn.ModuleList()
        self.params = {
            'module_list': ['Skip',
                            'ConvBn',
                            'ConvBn',
                            'SepConvBN',
                            'SepConvBN'],
            'name_list': ['Skip',
                          'ConvBn_k3',
                          'ConvBn_k5',
                          'SepConvBN_k3',
                          'SepConvBN_k5'],
            'Skip': {'in_chan': channles,
                     'out_chan': out_channels,
                     'reduction': False},
            'ConvBn_k3': {'in_chan': channles,
                          'k_size': 3,
                          'out_chan': out_channels,
                          'relu': True,
                          'stride': 1},
            'ConvBn_k5': {'in_chan': channles,
                          'k_size': 5,
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


class Cifar10ReductionCellBlock(NetworkBlock):
    n_layers = 3
    n_comp_steps = 1

    def __init__(self, channles, out_channels):
        super(Cifar10ReductionCellBlock, self).__init__()
        self.channles = channles
        self.op_list = nn.ModuleList()
        self.params = {
            'module_list': ['ConvBn',
                            'ConvBn',
                            'SepConvBN',
                            'SepConvBN',
                            'SepConvBN'],
            'name_list': ['ConvBn_3',
                          'ConvBn_5',
                          'SepConvBN_3',
                          'SepConvBN_5',
                          'SepConvBN_7'],
            'ConvBn_3': {'in_chan': channles,
                          'k_size': 3,
                          'out_chan': out_channels,
                          'relu': True,
                          'stride': 2},
            'ConvBn_5': {'in_chan': channles,
                          'k_size': 5,
                          'out_chan': out_channels,
                          'relu': True,
                          'stride': 2},
            'SepConvBN_3': {'in_chan': channles,
                             'k_size': 3,
                             'out_chan': out_channels,
                             'relu': True,
                             'stride': 2},
            'SepConvBN_5': {'in_chan': channles,
                             'k_size': 5,
                             'out_chan': out_channels,
                             'relu': True,
                             'stride': 2},
            'SepConvBN_7': {'in_chan': channles,
                             'k_size': 7,
                             'out_chan': out_channels,
                             'relu': True,
                             'stride': 2},
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
        cost_list = []
        for i in range(len(self.op_list)):
            cost_list.append(self.op_list[i].get_flop_cost(x)[1])

        return cost_list

    def get_latency(self, x):
        cost_list = []
        for i in range(len(self.op_list)):
            cost_list.append(self.op_list[i].get_latency(x)[1])

        return cost_list

    def get_param_num(self, x):
        cost_list = []
        for i in range(len(self.op_list)):
            cost_list.append(self.op_list[i].get_param_num(x)[1])

        return cost_list


class PKCifar10SN(UniformSamplingSuperNetwork):
    def __init__(self,
                 blocks_per_stage,
                 cells_per_block,
                 channels_per_block,
                 data_prop,
                 static_proba, *args, **kwargs):
        super(PKCifar10SN, self).__init__(*args, **kwargs)
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
        head = ConvBn(self.in_chan, channels_per_block[0][0], k_size=3, stride=1, relu=True)
        # tail (固定计算节点，结构不可学习)
        tail = kwargs['out_layer']

        # search space（stage - block - cell）
        self.sbca = StageBlockCellArc(Cifar10CellBlock, Cifar10ReductionCellBlock, AddBlock, ConvBn, self.graph, is_cell_dense=True, is_block_dense=True)
        in_name, out_name =\
            self.sbca.generate(head,
                               tail,
                               blocks_per_stage,
                               cells_per_block,
                               channels_per_block)
        self.sampling_parameters = self.sbca.sampling_parameters
        self.blocks = self.sbca.blocks

        # set graph
        self.set_graph(self.graph, in_name, out_name)

    def loss(self, predictions, labels):
        return self._loss(predictions, labels)

    def accuray(self, predictions, labels):
        return self._accuracy_evaluator.accuracy(predictions, labels)

    def hierarchical(self):
        return self.sbca.hierarchical
