# -*- coding: UTF-8 -*-
# @Time    : 2020-04-07 17:14
# @File    : FrozenFixedNetwork.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from nas.networks.FixedNetwork import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from nas.component.NetworkBlock import *
from nas.component.NetworkCell import *
from nas.component.Loss import *
from nas.component.ClassificationAccuracyEvaluator import *
import networkx as nx
import copy


class FrozenFixedNetwork(FixedNetwork):
    def __init__(self, *args, **kwargs):
        super(FrozenFixedNetwork, self).__init__(*args, **kwargs)

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

            self.graph.node[node]['out'] = out

            if node == self.out_node:
                model_out = out
                break

            # 3.3.step set successor input
            for succ in self.graph.successors(node):
                if succ not in data_dict:
                    data_dict[succ] = []

                data_dict[succ].append(out)

        return model_out

    def output(self, node_name):
        return self.graph.node[node_name]['out']