# -*- coding: UTF-8 -*-
# @Time    : 2020-04-07 17:14
# @File    : FrozenFixedNetwork.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antnas.networks.FixedNetwork import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from antnas.component.NetworkBlock import *
from antnas.component.NetworkCell import *
from antnas.component.Loss import *
from antnas.component.ClassificationAccuracyEvaluator import *
import networkx as nx
import copy


class FrozenFixedNetwork(FixedNetwork):
    def __init__(self, *args, **kwargs):
        super(FrozenFixedNetwork, self).__init__(*args, **kwargs)
        NetworkBlock.bn_track_running_stats = False
        NetworkBlock.bn_moving_momentum = False

    def forward(self, x, y):
        # 1.step parse x,y - (data,label)
        input = [x]

        data_dict = {}
        data_dict[self.in_node] = [*input]

        node_output_dict = {}
        for node in self.traversal_order:
            cur_node = self.graph.node[node]
            input = self.format_input(data_dict[node])

            # 3.2.step execute node op
            out = self.blocks[cur_node['module']](input)
            node_output_dict[node] = out

            if node == self.out_node:
                break

            # 3.3.step set successor input
            for succ in self.graph.successors(node):
                if succ not in data_dict:
                    data_dict[succ] = []

                data_dict[succ].append(out)

        return node_output_dict