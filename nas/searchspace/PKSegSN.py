# -*- coding: UTF-8 -*-
# @Time    : 2020-05-17 18:23
# @File    : PKSegSN.py
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
from nas.component.SegmentationAccuracyEvaluator import *
from nas.searchspace.PKMixArc import *


__all__ = ['PKSegSN']


class SegDecoderCellBlock(NetworkBlock):
    n_layers = 3
    n_comp_steps = 1

    def __init__(self, channles, out_channels, reduction=False):
        super(SegDecoderCellBlock, self).__init__()
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


class SegDecoderASPPCell(NetworkBlock):
    n_layers = 3
    n_comp_steps = 1

    def __init__(self, in_chan, depth):
        super(SegDecoderASPPCell, self).__init__()
        
        self.op_list = nn.ModuleList()
        self.in_chan = in_chan
        self.depth = depth
        
        self.params = {
            'module_list': ['ASPPBlock',
                            'ASPPBlock',
                            'ASPPBlock',
                            'ASPPBlock',
                            'ASPPBlock'],
            'name_list': ['aspp_dilation_246',
                          'aspp_dilation_357',
                          'aspp_dilation_257',
                          'aspp_dilation_235',
                          'aspp_dilation_347'],
            'aspp_dilation_246':{'in_chan': in_chan,
                                 'depth': depth,
                                 'atrous_rates': [2, 4, 6]},
            'aspp_dilation_357':{'in_chan': in_chan,
                                 'depth': depth,
                                 'atrous_rates': [3, 5, 7]},
            'aspp_dilation_257':{'in_chan': in_chan,
                                 'depth': depth,
                                 'atrous_rates': [2, 5, 7]},
            'aspp_dilation_235':{'in_chan': in_chan,
                                 'depth': depth,
                                 'atrous_rates': [2, 3, 5]},
            'aspp_dilation_347':{'in_chan': in_chan,
                                 'depth': depth,
                                 'atrous_rates': [3, 4, 7]},
            'in_chan': in_chan,
            'out_chan': depth
        }
        self.structure_fixed = True
        self.op_list = self.build()
        
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


class PKSegSN(UniformSamplingSuperNetwork):
    def __init__(self, data_prop, *args, **kwargs):
        super(PKSegSN, self).__init__(*args, **kwargs)
        
        self.in_chan = data_prop['in_channels']
        self.in_size = data_prop['img_dim']
        self.out_dim = data_prop['out_size'][0]

        self.graph = nx.DiGraph()
        self.sampling_parameters = nn.ParameterList()
        
        self._loss = cross_entropy
        self._accuracy_evaluator = SegmentationAccuracyEvaluator(class_num=self.out_dim)

        # head (固定计算节点，对应激活参数不可学习)
        head = ConvBn(self.in_chan, 32, k_size=3, stride=2, relu=True)
        # tail (固定计算节点，结构不可学习)
        tail = kwargs['out_layer']
        
        self.seg_arc = PKMixArc(SegDecoderCellBlock,
                                SegDecoderASPPCell,
                                AddBlock,
                                ConvBn,
                                self.graph)
        in_name, out_name = self.seg_arc.generate(head, tail)
        self.blocks = self.seg_arc.blocks
        
        # set graph
        self.set_graph(self.graph, in_name, out_name)
    
    def loss(self, predictions, labels):
        return self._loss(predictions, labels)
    
    def accuray(self, predictions, labels):
        return self._accuracy_evaluator.accuracy(predictions, labels)
    
    def hierarchical(self):
        return self.seg_arc.hierarchical