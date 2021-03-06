# -*- coding: UTF-8 -*-
# @Time    : 2020-04-20 09:04
# @File    : Arc.py.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function


from antnas.networks.SuperNetwork import *
from antnas.utils.drawers.NASDrawer import NASDrawer
import torch
import torch.nn as nn


class Arc(object):
    def __init__(self, graph=None, blocks=None):
        self.graph = graph
        if self.graph is None:
            self.graph = nx.DiGraph()

        self.blocks = blocks
        if self.blocks is None:
            self.blocks = nn.ModuleList([])
        self.sampling_parameters = None

        self._in_node = ""
        self._out_node = ""

        self.path_recorder = None
        self.traversal_order = None
        self._offset = 0

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
    
    @property
    def offset(self):
        return self._offset
    
    @offset.setter
    def offset(self, val):
        self._offset = val

    @property
    def net(self):
        return self.graph

    def arc_loss(self, shape, loss='latency', latency_lookup_table=None, devices=[]):
        if len(devices) > 1:
            NetworkBlock.device_num = len(devices)

        # 创建统计结构损失对象
        assert(loss in ['comp', 'latency', 'param'])
        if loss == 'comp':
            self.cost_evaluator = ComputationalCostEvaluator(model=self, main_cost=False)
        elif loss == 'latency':
            self.cost_evaluator = LatencyCostEvaluator(model=self, main_cost=False, latency=latency_lookup_table)
        else:
            self.cost_evaluator = ParameterCostEvaluator(model=self, main_cost=False)

        # 统计结构损失
        self.traversal_order = list(nx.topological_sort(self.graph))
        self.path_recorder = PathRecorder(self.graph, self.out_node)

        sampling = torch.Tensor()
        active = torch.Tensor()

        for node_name in self.traversal_order:
            cur_node = self.graph.node[node_name]
            sampled_v = cur_node['sampled']

            structure_fixed = False
            if node_name.startswith('CELL'):
                structure_fixed = True

            sampling, active = \
                self.path_recorder.add_sampling(node_name,
                                                torch.as_tensor([sampled_v]).reshape([1, 1, 1, 1]),
                                                sampling,
                                                active,
                                                structure_fixed)

        # 初始化结构损失估计函数
        self.cost_evaluator.init_costs(self, self.graph, input_node=self.in_node, input_shape=shape)

        # 获得结构损失
        sampled_arc, pruned_arc = \
            self.path_recorder.get_arch(self.out_node, sampling, active)
        sampled_cost, pruned_cost = \
            self.cost_evaluator.get_costs([sampled_arc, pruned_arc],
                                          device=None if len(devices) <= 1 else [i for i in range(len(devices))])

        return sampled_cost, pruned_cost
    
    def arc_min_max(self, shape, loss='latency'):
        # find max/min arch
        max_loss = -1
        min_loss = -1
        try_times = 1000
        traversal_order = list(nx.topological_sort(self.graph))

        while try_times > 0:
            feature = [None for _ in range(len(traversal_order))]
            for node_name in traversal_order:
                cur_node = self.graph.node[node_name]
                if not (node_name.startswith('CELL') or node_name.startswith('T')):
                    # 不可学习，处于永远激活状态
                    feature[cur_node['sampling_param']] = int(1)
                else:
                    if not self.blocks[cur_node['module']].structure_fixed:
                        feature[cur_node['sampling_param']] = int(np.random.randint(0, 2))
                    else:
                        feature[cur_node['sampling_param']] = int(np.random.randint(0, NetworkBlock.state_num))
        
            _, pruned_cost = \
                self.arc_loss(shape, loss=loss, feature=feature)
        
            if max_loss < pruned_cost or max_loss < 0:
                max_loss = pruned_cost.item()
        
            if min_loss > pruned_cost or min_loss < 0:
                min_loss = pruned_cost.item()
        
            try_times -= 1
    
        return min_loss, max_loss