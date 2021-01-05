# -*- coding: UTF-8 -*-
# @Time    : 2020-05-25 10:17
# @File    : PKImageNetSN.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function


import networkx as nx
from antnas.component.NetworkCell import *
from antnas.networks.UniformSamplingSuperNetwork import UniformSamplingSuperNetwork
from antnas.utils.drawers.NASDrawer import NASDrawer
from antnas.component.ClassificationAccuracyEvaluator import *
from antnas.searchspace.DualStageBlockCellArc import *
from antnas.networks.ContinuousSuperNetwork import ContinuousSuperNetwork

__all__ = ['PKAsynImageNetContinuousSN']


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
                          'IRB_k3e3_nohs',
                          'IRB_k3e3_nose',
                          'IRB_k5e3_nohs',
                          'IRB_k5e3_nose'],
            'Skip': {'in_chan': channles,
                     'out_chan': out_channels,
                     'reduction': False},
            'IRB_k3e3_nohs': {'in_chan': channles,
                              'kernel_size': 3,
                              'expansion': 3,
                              'out_chan': out_channels,
                              'reduction': False,
                              'skip': True,
                              'ratio': 4,
                              'hs': False,
                              'se': True},
            'IRB_k3e3_nose': {'in_chan': channles,
                                   'kernel_size': 3,
                                   'expansion': 3,
                                   'out_chan': out_channels,
                                   'reduction': False,
                                   'skip': True,
                                   'ratio': 4,
                                   'hs': True,
                                   'se': False},
            'IRB_k5e3_nohs': {'in_chan': channles,
                         'kernel_size': 5,
                         'expansion': 3,
                         'out_chan': out_channels,
                         'reduction': False,
                         'skip': True,
                         'ratio': 4,
                         'hs': False,
                         'se': True},
            'IRB_k5e3_nose': {'in_chan': channles,
                              'kernel_size': 5,
                              'expansion': 3,
                              'out_chan': out_channels,
                              'reduction': False,
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
        cost_list = []
        if NetworkBlock.device_num == 1:
            for i in range(len(self.op_list)):
                cost_list.append(self.op_list[i].get_latency(x)[1])
        else:
            cost_list = [[], []]
            for i in range(len(self.op_list)):
                cost_list[0].append(self.op_list[i].get_latency(x)[0][1])
                cost_list[1].append(self.op_list[i].get_latency(x)[1][1])
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
            'name_list': ['IRB_k3e3_nose',
                          'IRB_k3e3_nohs',
                          'IRB_k3e3_nohs_nose',
                          'IRB_k3e3',
                          'IRB_k5e3_nohs_nose'],
            'IRB_k3e3_nose': {'in_chan': channles,
                              'kernel_size': 3,
                              'expansion': 3,
                              'out_chan': out_channels,
                              'reduction': True,
                              'skip': False,
                              'ratio': 4,
                              'hs': True,
                              'se': False},
            'IRB_k3e3_nohs': {'in_chan': channles,
                              'kernel_size': 3,
                              'expansion': 3,
                              'out_chan': out_channels,
                              'reduction': True,
                              'skip': False,
                              'ratio': 4,
                              'hs': False,
                              'se': True},
            'IRB_k3e3_nohs_nose': {'in_chan': channles,
                                   'kernel_size': 3,
                                   'expansion': 3,
                                   'out_chan': out_channels,
                                   'reduction': True,
                                   'skip': False,
                                   'ratio': 4,
                                   'hs': False,
                                   'se': False},
            'IRB_k3e3': {'in_chan': channles,
                         'kernel_size': 3,
                         'expansion': 3,
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
        if NetworkBlock.device_num == 1:
            for i in range(len(self.op_list)):
                cost_list.append(self.op_list[i].get_latency(x)[1])
        else:
            cost_list = [[], []]
            for i in range(len(self.op_list)):
                cost_list[0].append(self.op_list[i].get_latency(x)[0][1])
                cost_list[1].append(self.op_list[i].get_latency(x)[1][1])
        return cost_list

    def get_param_num(self, x):
        cost_list = []
        for i in range(len(self.op_list)):
            cost_list.append(self.op_list[i].get_param_num(x)[1])

        return cost_list


class PKAsynImageNetContinuousSN(ContinuousSuperNetwork):
    def __init__(self,
                 blocks_per_stage,
                 cells_per_block,
                 channels_per_block,
                 data_prop,
                 *args, **kwargs):
        super(PKAsynImageNetContinuousSN, self).__init__(*args, **kwargs)
        NetworkBlock.state_num = 5
        self.in_chan = data_prop['in_channels']
        self.in_size = data_prop['img_dim']
        self.out_dim = data_prop['out_size'][0]
        self.disturb_ratio = 0.2

        self._input_size = (self.in_chan, self.in_size, self.in_size)
        self.graph = nx.DiGraph()
        self.sampling_parameters = nn.ParameterList()
        self._loss = nn.CrossEntropyLoss()

        # head (固定计算节点，对应激活参数不可学习)
        head = ConvBn(self.in_chan, channels_per_block[0][0], k_size=3, stride=2, relu=True)
        # tail (固定计算节点，结构不可学习)
        tail = kwargs['out_layer']

        # search space（stage - block - cell）
        self.sbca = \
            DualStageBlockCellArc(ImageNetCellBlock,
                              ImageNetReductionCellBlock,
                              ConcatBlock,
                              ConvBn,
                              self.graph,
                              cross_interval=1)
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

        # 损失函数
        self._criterion = nn.CrossEntropyLoss()

        # 保存搜索空间图
        a = NASDrawer()
        a.draw(self.graph, filename='./searchspace.svg')

    @property
    def criterion(self):
        return self._criterion

    @property
    def accuracy_evaluator(self):
        return ClassificationAccuracyEvaluator

    def hierarchical(self):
        return self.sbca.hierarchical
