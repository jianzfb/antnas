# -*- coding: UTF-8 -*-
# @Time    : 2019-09-20 14:08
# @File    : FabricSegSN.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from nas.interfaces.NetworkBlock import *
from nas.interfaces.NetworkCell import *
from nas.networks.StochasticSuperNetwork import StochasticSuperNetwork

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

__all__ = ['ConvolutionalNeuralFabric']


def downsampling_layer(n_chan, k_size):
    return InvertedResidualBlock(n_chan,
                                 expansion=3,
                                 kernel_size=k_size,
                                 out_chan=n_chan,
                                 skip=False,
                                 reduction=True)


def samesampling_layer(n_chan, k_size):
    return InvertedResidualBlock(n_chan,
                                 expansion=3,
                                 kernel_size=k_size,
                                 out_chan=n_chan,
                                 skip=True,
                                 reduction=False)


def upsampling_layer(n_chan, k_size):
    return ResizedBlock(n_chan, n_chan, True, k_size, 2)


class Out_Layer(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, in_chan, out_shape, bias=True):
        super(Out_Layer, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_shape[0], 1, bias=bias)
        self.out_shape = out_shape

        self.params = {
            'module_list': ['Out_Layer'],
            'name_list': ['Out_Layer'],
            'Out_Layer': {'out_shape': out_shape},
            'out': 'outname'
        }

    def forward(self, x):
        x = self.conv(x)
        return x.view(-1, *self.out_shape)

    def get_flop_cost(self, x):
        return [0] * NetworkBlock.state_num


class ConvolutionalNeuralFabric(StochasticSuperNetwork):
    _NODE_FORMAT = 'A-{}_{}'
    _TRANSFORM_FORMAT = 'T-{}_{}-{}_{}'

    def __init__(self,
                 n_layer,
                 n_chan,
                 data_prop,
                 kernel_size=3,
                 *args, **kwargs):
        super(ConvolutionalNeuralFabric, self).__init__(*args, **kwargs)
        self.n_layer = n_layer
        self.n_chan = n_chan
        self.n_scale = int(np.log2(data_prop['img_dim']) + 1)
        self.in_size = data_prop['img_dim']
        self.in_chan = data_prop['in_channels']
        self.out_size = data_prop['out_size']
        self.out_dim = len(data_prop['out_size'])
        self.kernel_size = kernel_size

        # set Block state number
        NetworkBlock.state_num = 2

        self._input_size = (self.in_chan, self.in_size, self.in_size)

        conv_params = (self.n_chan, self.kernel_size)
        self.downsampling = lambda: downsampling_layer(*conv_params)
        self.samesampling = lambda: samesampling_layer(*conv_params)
        self.upsampling = lambda: upsampling_layer(*conv_params)

        self._loss = cross_entropy
        self._accuracy_evaluator = SegmentationAccuracyEvaluator(class_num=self.out_dim)

        self.blocks = nn.ModuleList([])
        self.graph = nx.DiGraph()
        self.sampling_parameters = nn.ParameterList()
        self.is_classif = False

        # first layer
        in_name = self.init_first_layer()

        for i in range(1, self.n_layer):
            self.add_layer(i)

        if self.n_layer > 1:
            # Add vertical connections in the last layers :
            self.add_zip_layer(self.n_layer - 1, down=False)

        out_name = self.add_output_layer()
        self.set_graph(self.graph, in_name, out_name)

    def init_first_layer(self):
        position = (0, 0)
        in_module = ConvBn(self.in_chan, self.n_chan, relu=True)
        input_node = self.add_aggregation(position, module=in_module)

        for j in range(1, self.n_scale):
            position = (0, j)
            self.add_aggregation(position, module=AddBlock())

        self.add_zip_layer(0)
        return input_node

    def add_layer(self, layer):
        in_layer = layer - 1

        for scale in range(self.n_scale):
            self.add_aggregation((layer, scale), AddBlock())

            min_scale = np.max([0, scale - 1])
            max_scale = np.min([self.n_scale, scale + 2])
            for k in range(min_scale, max_scale):
                if k < scale:  # The input has a finer scale -> downsampling
                    module = self.downsampling()
                if k == scale:  # The input has the same scale -> samesampling
                    module = self.samesampling()
                if k > scale:  # The input has a coarser scale -> upsampling
                    module = self.upsampling()

                self.add_transformation((in_layer, k), (layer, scale), module)

    def add_zip_layer(self, layer, down=True):
        module = self.downsampling if down else self.upsampling

        for j in range(0, self.n_scale - 1):
            node1 = (layer, j)
            node2 = (layer, j + 1)
            src, dst = (node1, node2) if down else (node2, node1)
            self.add_transformation(src, dst, module())

    def add_output_layer(self):
        last_layer = self.n_layer - 1
        out_scale = (self.n_scale - 1) if self.is_classif else 0

        out_features_name = self._NODE_FORMAT.format(last_layer, out_scale)

        out_pos = (last_layer + 1, out_scale)
        out_name = 'Lin-{}_{}-out'.format(*out_pos)

        out_module = Out_Layer(self.n_chan, self.out_size, True)

        sampling_param = self.sampling_param_generator(out_name)

        self.graph.add_node(out_name,
                            module=len(self.blocks),
                            module_params=out_module.params,
                            sampling_param=len(self.sampling_parameters),
                            pos=BSNDrawer.get_draw_pos(pos=out_pos))
        self.graph.add_edge(out_features_name, out_name, width_node=out_name)

        if sampling_param is not None:
            self.sampling_parameters.append(sampling_param)
        self.blocks.append(out_module)

        return out_name

    def add_transformation(self, source, dest, module):
        src_l, src_s = source
        dst_l, dst_s = dest

        trans_name = self._TRANSFORM_FORMAT.format(src_l, src_s, dst_l, dst_s)
        source_name = self._NODE_FORMAT.format(src_l, src_s)
        dest_name = self._NODE_FORMAT.format(dst_l, dst_s)

        pos = BSNDrawer.get_draw_pos(source=source, dest=dest)

        sampling_param = self.sampling_param_generator(trans_name)

        self.graph.add_node(trans_name,
                            module=len(self.blocks),
                            module_params=module.params,
                            sampling_param=len(self.sampling_parameters),
                            pos=pos)
        self.graph.add_edge(source_name, trans_name, width_node=trans_name)
        self.graph.add_edge(trans_name, dest_name, width_node=trans_name)

        self.sampling_parameters.append(sampling_param)
        self.blocks.append(module)
        return trans_name

    def add_aggregation(self, pos, module):
        agg_node_name = self._NODE_FORMAT.format(*pos)
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

    def loss(self, predictions, labels):
        return self._loss(predictions, labels)

    def accuray(self, predictions, labels):
        return self._accuracy_evaluator.accuracy(predictions, labels)
