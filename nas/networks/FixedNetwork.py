# -*- coding: UTF-8 -*-
# @Time    : 2020-03-30 18:12
# @File    : FixedNetwork.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from nas.component.NetworkBlock import *
from nas.component.NetworkCell import *
from nas.component.Loss import *
from nas.component.ClassificationAccuracyEvaluator import *
import networkx as nx
import copy
from nas.utils.drawers.NASDrawer import *


class FixedNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super(FixedNetwork, self).__init__()
        NetworkBlock.bn_track_running_stats = True
        NetworkBlock.bn_moving_momentum = True
        
        architecture_path = kwargs.get('architecture', None)
        self.output_layer_cls = kwargs.get('output_layer_cls', None)
        self.graph = nx.read_gpickle(architecture_path)

        # self._loss = cross_entropy
        self._loss = nn.CrossEntropyLoss()
        self._accuracy_evaluator = ClassificationAccuracyEvaluator()

        self.in_node = None
        self.out_node = None
        # traverse all nodes in graph
        self.traversal_order = list(nx.topological_sort(self.graph))
        self.blocks = nn.ModuleList([None for _ in range(len(self.traversal_order))])
        for node_index, node_name in enumerate(self.traversal_order):
            if node_index == 0:
                self.in_node = node_name
            if node_index == len(self.traversal_order) - 1:
                self.out_node = node_name

            cur_node = self.graph.node[node_name]

            module_list = cur_node['module_params']['module_list']
            sampled_module_index = cur_node['sampled']
            sampled_module = None
            sampled_module_name = ''
            if len(module_list) == 1:
                if sampled_module_index == 1:
                    if node_index == len(self.traversal_order) - 1:
                        sampled_module = self.output_layer_cls
                    else:
                        sampled_module = globals()[module_list[0]]

                    sampled_module_name = cur_node['module_params']['name_list'][0]
                    self.blocks[cur_node['module']] = \
                        sampled_module(**cur_node['module_params'][sampled_module_name])
                else:
                    sampled_module_name = cur_node['module_params']['name_list'][0]
                    self.blocks[cur_node['module']] = Zero(**cur_node['module_params'][sampled_module_name])
                    sampled_module_name = 'ZERO'
            else:
                sampled_module = globals()[module_list[sampled_module_index]]
                sampled_module_name = cur_node['module_params']['name_list'][sampled_module_index]
                self.blocks[cur_node['module']] = \
                    sampled_module(**cur_node['module_params'][sampled_module_name])

            print('build node %s with sampling %s op' % (node_name, sampled_module_name))
            
        self.plotter = kwargs.get('plotter', None)
        if self.plotter is not None:
            self.drawer = NASDrawer(self.plotter)
            self.drawer.draw(self.graph)
        
    def forward(self, x, y):
        # 1.step parse x,y - (data,label)
        input = [x]

        # running_graph.node[self.in_node]['input'] = [*input]
        data_dict = {}
        data_dict[self.in_node] = [*input]

        model_out = None
        for node in self.traversal_order:
            cur_node = self.graph.node[node]
            input = self.format_input(data_dict[node])

            # 3.2.step execute node op
            out = self.blocks[cur_node['module']](input)

            if node == self.out_node:
                model_out = out
                break

            # 3.3.step set successor input
            for succ in self.graph.successors(node):
                if succ not in data_dict:
                    data_dict[succ] = []

                data_dict[succ].append(out)

        # 4.step compute model loss
        indiv_loss = self.loss(model_out, y)

        # 5.step compute model accuracy
        model_accuracy = self.accuray(model_out, y)

        # 6.step total loss
        return indiv_loss, model_accuracy

    def __str__(self):
        return ''

    @staticmethod
    def format_input(input):
        if (isinstance(input, tuple) or isinstance(input, list)) and len(input) == 1:
            input = input[0]
        return input

    def loss(self, predictions, labels):
        return self._loss(predictions, labels)

    def accuray(self, predictions, labels):
        return self._accuracy_evaluator.accuracy(predictions, labels)

