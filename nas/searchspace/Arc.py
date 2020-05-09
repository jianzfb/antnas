# -*- coding: UTF-8 -*-
# @Time    : 2020-04-20 09:04
# @File    : Arc.py.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function


from nas.networks.SuperNetwork import *
from nas.utils.drawers.NASDrawer import NASDrawer
import torch
import torch.nn as nn


class Arc:
    def __init__(self, graph=None):
        self.graph = graph
        if self.graph is None:
            self.graph = nx.DiGraph()

        self.blocks = nn.ModuleList([])
        self.sampling_parameters = None

        self._in_node = ""
        self._out_node = ""

        kk = {
            'comp': ComputationalCostEvaluator,
            'latency': LatencyCostEvaluator,
            'param': ParameterCostEvaluator
        }

        self.cost_evaluators = {}
        for k, v in kk.items():
            self.cost_evaluators[k] = v(model=self, main_cost=False)

        self.path_recorder = None
        self.traversal_order = None

    @property
    def in_node(self):
        return self._in_node

    @in_node.setter
    def in_node(self, val):
        self._in_node = val

    @property
    def out_node(self):
        return self._out_node

    @out_node.setter
    def out_node(self, val):
        self._out_node = val

    @property
    def arch_node_index(self):
        return self.path_recorder.node_index

    @staticmethod
    def format_input(input):
        if (isinstance(input, tuple) or isinstance(input, list)) and len(input) == 1:
            input = input[0]
        return input

    def arc_loss(self, shape, loss='latency'):
        x = torch.ones(shape)
        self.traversal_order = list(nx.topological_sort(self.graph))
        self.path_recorder = PathRecorder(self.graph, self.out_node)

        sampling = torch.Tensor()
        active = torch.Tensor()

        # 初始化path_recorder
        feature = [1 for _ in range(len(self.traversal_order))]
        for node_name in self.traversal_order:
            cur_node = self.graph.node[node_name]

            sampling, active = \
                self.path_recorder.add_sampling(node_name,
                                                torch.as_tensor([feature[cur_node['sampling_param']]]).reshape(
                                                    [1, 1, 1, 1]),
                                                sampling,
                                                active,
                                                self.blocks[cur_node['module']].structure_fixed)

        # 初始化每一节点数据形状
        for node in self.traversal_order:
            self.graph.node[self.in_node]['input'] = [x]
            cur_node = self.graph.node[node]
            input = self.format_input(cur_node['input'])

            if len(input) == 0:
                raise RuntimeError('Node {} has no inputs'.format(node))
            print(node)
            out = self.blocks[cur_node['module']](input)
            if node == self.out_node:
                break

            # 3.3.step set successor input
            for succ in self.graph.successors(node):
                if 'input' not in self.graph.node[succ]:
                    self.graph.node[succ]['input'] = []
                self.graph.node[succ]['input'].append(out)

        # 初始化结构损失估计函数
        self.cost_evaluators[loss].init_costs(self, self.graph)

        # 获得结构损失
        sampled_arc, pruned_arc = \
            self.path_recorder.get_arch(self.out_node, sampling, active)
        sampled_cost, pruned_cost = \
            self.cost_evaluators[loss].get_costs([sampled_arc, pruned_arc])

        return sampled_cost, pruned_cost