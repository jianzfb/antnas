# -*- coding: UTF-8 -*-
# @Time    : 2020-05-25 10:17
# @File    : PKImageNetSN.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function


import networkx as nx
from nas.component.NetworkCell import *
from nas.networks.UniformSamplingSuperNetwork import UniformSamplingSuperNetwork
from nas.utils.drawers.NASDrawer import NASDrawer
from nas.component.Loss import *
from nas.component.ClassificationAccuracyEvaluator import *
from nas.searchspace.StageBlockCellArc import *

__all__ = ['PKImageNetSN']


class ImageNetCellBlock(NetworkBlock):
    n_layers = 3
    n_comp_steps = 1

    def __init__(self, channles, out_channels, reduction=False):
        super(ImageNetCellBlock, self).__init__()
        self.channles = channles
        self.op_list = nn.ModuleList()
        self.params = {
            'module_list': ['Skip',
                            'InvertedResidualBlockWithSEHS',
                            'InvertedResidualBlockWithSEHS',
                            'InvertedResidualBlockWithSEHS',
                            'InvertedResidualBlockWithSEHS'],
            'name_list': ['Skip',
                          'IRB_k3e6_nohs',
                          'IRB_k3e6_nose',
                          'IRB_k5e3_nohs',
                          'IRB_k5e3_nose'],
            'Skip': {'in_chan': channles,
                     'out_chan': out_channels,
                     'reduction': reduction},
            'IRB_k3e6_nohs': {'in_chan': channles,
                              'kernel_size': 3,
                              'expansion': 6,
                              'out_chan': out_channels,
                              'reduction': reduction,
                              'skip': True,
                              'ratio': 4,
                              'hs': False,
                              'se': True},
            'IRB_k3e6_nose': {'in_chan': channles,
                                   'kernel_size': 3,
                                   'expansion': 6,
                                   'out_chan': out_channels,
                                   'reduction': reduction,
                                   'skip': True,
                                   'ratio': 4,
                                   'hs': True,
                                   'se': False},
            'IRB_k5e3_nohs': {'in_chan': channles,
                         'kernel_size': 5,
                         'expansion': 3,
                         'out_chan': out_channels,
                         'reduction': reduction,
                         'skip': True,
                         'ratio': 4,
                         'hs': False,
                         'se': True},
            'IRB_k5e3_nose': {'in_chan': channles,
                              'kernel_size': 5,
                              'expansion': 3,
                              'out_chan': out_channels,
                              'reduction': reduction,
                              'skip': True,
                              'ratio': 4,
                              'hs': True,
                              'se': False},
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


class ImageNetReductionCellBlock(NetworkBlock):
    n_layers = 3
    n_comp_steps = 1

    def __init__(self, channles, out_channels):
        super(ImageNetReductionCellBlock, self).__init__()
        self.channles = channles
        self.op_list = nn.ModuleList()
        self.params = {
            'module_list': ['InvertedResidualBlockWithSEHS',
                            'InvertedResidualBlockWithSEHS',
                            'InvertedResidualBlockWithSEHS',
                            'InvertedResidualBlockWithSEHS',
                            'InvertedResidualBlockWithSEHS'],
            'name_list': ['IRB_k3e6_nose',
                          'IRB_k3e6_nohs',
                          'IRB_k3e6_nohs_nose',
                          'IRB_k3e6',
                          'IRB_k5e3_nohs_nose'],
            'IRB_k3e6_nose': {'in_chan': channles,
                              'kernel_size': 3,
                              'expansion': 6,
                              'out_chan': out_channels,
                              'reduction': True,
                              'skip': False,
                              'ratio': 4,
                              'hs': True,
                              'se': False},
            'IRB_k3e6_nohs': {'in_chan': channles,
                              'kernel_size': 3,
                              'expansion': 6,
                              'out_chan': out_channels,
                              'reduction': True,
                              'skip': False,
                              'ratio': 4,
                              'hs': False,
                              'se': True},
            'IRB_k3e6_nohs_nose': {'in_chan': channles,
                                   'kernel_size': 3,
                                   'expansion': 6,
                                   'out_chan': out_channels,
                                   'reduction': True,
                                   'skip': False,
                                   'ratio': 4,
                                   'hs': False,
                                   'se': False},
            'IRB_k3e6': {'in_chan': channles,
                         'kernel_size': 3,
                         'expansion': 6,
                         'out_chan': out_channels,
                         'reduction': True,
                         'skip': False,
                         'ratio': 4,
                         'hs': True,
                         'se': True},
            'IRB_k5e3_nohs_nose': {'in_chan': channles,
                              'kernel_size': 5,
                              'expansion': 3,
                              'out_chan': out_channels,
                              'reduction': True,
                              'skip': False,
                              'ratio': 4,
                              'hs': False,
                              'se': False},
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


class PKImageNetSN(UniformSamplingSuperNetwork):
    def __init__(self,
                 blocks_per_stage,
                 cells_per_block,
                 channels_per_block,
                 data_prop,
                 static_proba, *args, **kwargs):
        super(PKImageNetSN, self).__init__(*args, **kwargs)
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
        head = ConvBn(self.in_chan, channels_per_block[0][0], k_size=3, stride=2, relu=True)
        # tail (固定计算节点，结构不可学习)
        tail = kwargs['out_layer']

        # search space（stage - block - cell）
        self.sbca = \
            StageBlockCellArc(ImageNetCellBlock,
                              ImageNetReductionCellBlock,
                              AddBlock,
                              ConvBn,
                              self.graph,
                              is_cell_dense=True,
                              is_block_dense=True)
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
