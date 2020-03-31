# -*- coding: UTF-8 -*-
# @Time    : 2019-07-24 11:39
# @File    : BaselineSN.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import networkx as nx
from nas.component.NetworkCell import *
from nas.networks.UniformSamplingSuperNetwork import UniformSamplingSuperNetwork
from nas.utils.drawers.BSNDrawer import BSNDrawer
from nas.component.Loss import *
from nas.component.ClassificationAccuracyEvaluator import *
from nas.searchspace.StageBlockCellArc import *

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
            'OutLayer': {'out_shape': out_shape, 'in_chan': in_chan},
            'out': 'outname'
        }

    def forward(self, x, sampling=None):
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


class BaselineSN(UniformSamplingSuperNetwork):
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
        head = ConvBn(self.in_chan, channels_per_block[0][0], k_size=3, stride=2, relu=True)
        # tail (固定计算节点，结构不可学习)
        tail = OutLayer(channels_per_block[-1][-1], data_prop['out_size'], True)

        # search space（stage - block - cell）
        self.sbca = StageBlockCellArc(CellBlock, ReductionCellBlock, AddBlock, ConvBn, self.graph)
        in_name, out_name =\
            self.sbca.generate(head,
                               tail,
                               channels_per_block[0][0],
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


