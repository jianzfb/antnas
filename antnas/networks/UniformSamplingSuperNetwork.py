# -*- coding: UTF-8 -*-
# @Time    : 2019-09-29 11:14
# @File    : UniformSamplingSuperNetwork.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from antnas.networks.SuperNetwork import SuperNetwork
from antnas.component.NetworkBlock import *
from antnas.component.NetworkCell import *
from antnas.component.PathRecorder import PathRecorder
from antnas.component.NetworkBlock import *
import copy
import networkx as nx


class UniformSamplingSuperNetwork(SuperNetwork):
    def __init__(self, *args, **kwargs):
        super(UniformSamplingSuperNetwork, self).__init__(*args, **kwargs)

    def forward(self, x, y, arc, epoch=None, warmup=False):
        # 1.step parse x,y - (data,label)
        input = [x]

        # 2.step get sampling architecture of
        # using same architecture in batch
        assert(arc is not None)
        batched_sampling = arc[0, :].view((1, arc.shape[1]))
        
        # 3.step forward network
        # 3.1.step set the input of network graph
        # running_graph.node[self.in_node]['input'] = [*input]
        data_dict = {}
        data_dict[self.in_node] = [*input]

        model_out = None
        for node in self.traversal_order:
            cur_node = self.net.node[node]
            # input = self.format_input(cur_node['input'])
            input = self.format_input(data_dict[node])

            if len(input) == 0:
                raise RuntimeError('Node {} has no inputs'.format(node))
            batch_size = input[0].size(0) if type(input) == list else input.size(0)
            node_sampling = self.get_node_sampling(node, 1, batched_sampling)

            # 3.2.step execute node op
            out = self.blocks[cur_node['module']](input, node_sampling.squeeze())

            if node == self.out_node:
                model_out = out
                break

            # 3.3.step set successor input
            for succ in self.graph.successors(node):
                if succ not in data_dict:
                    data_dict[succ] = []

                data_dict[succ].append(out)

        return None, model_out, None, None

    def search_and_save(self, folder=None, name=None):
        if not os.path.exists(folder):
            os.makedirs(folder)

        path = os.path.join(folder, name)
        # 1.step save supernet model
        torch.save(self.state_dict(), '%s.supernet.model'%path)

        # 2.step search parel front

    @property
    def n_layers(self):
        return sum([mod.n_layers for mod in self.blocks])

    @property
    def n_comp_steps(self):
        return sum([mod.n_comp_steps for mod in self.blocks])
    
    def sample_arch(self, *args, **kwargs):
        # get constraint condition
        comp_min = kwargs.get('comp_min', self.arch_objective_comp_min)
        comp_max = kwargs.get('comp_max', self.arch_objective_comp_max)
        latency_min = kwargs.get('latency_min', self.arch_objective_latency_min)
        latency_max = kwargs.get('latency_max', self.arch_objective_latency_max)
        param_min = kwargs.get('param_min', self.arch_objective_param_min)
        param_max = kwargs.get('param_max', self.arch_objective_param_max)
        
        # sampling satisfied feature
        sampling_feature = None
        max_try_count = 50
        try_count = 0
        with torch.no_grad():
            while try_count < max_try_count:
                feature = [None for _ in range(len(self.traversal_order))]

                sampling = torch.Tensor()
                active = torch.Tensor()
                for node_name in self.traversal_order:
                    cur_node = self.net.node[node_name]
                    if not (node_name.startswith('CELL') or node_name.startswith('T')):
                        # 不可学习，处于永远激活状态
                        feature[cur_node['sampling_param']] = int(1)
                    else:
                        if not self.blocks[cur_node['module']].structure_fixed:
                            feature[cur_node['sampling_param']] = int(np.random.randint(0, 2))
                        else:
                            feature[cur_node['sampling_param']] = int(np.random.randint(0, NetworkBlock.state_num))

                    sampling, active = \
                        self.path_recorder.add_sampling(node_name,
                                                        torch.as_tensor([feature[cur_node['sampling_param']]]).reshape([1,1,1,1]),
                                                        sampling,
                                                        active,
                                                        self.blocks[cur_node['module']].structure_fixed)
                satisfied_constraint = True
                for cost, cost_eval in self.arch_cost_evaluators.items():
                    sampled_arc, pruned_arc = \
                        self.path_recorder.get_arch(self.out_node, sampling, active)
                    sampled_cost, pruned_cost = \
                        cost_eval.get_costs([sampled_arc, pruned_arc])
    
                    if comp_max > 0 and comp_min > 0 and cost == "comp":
                        if pruned_cost > comp_max or pruned_cost < comp_min:
                            satisfied_constraint = False
                            break
    
                    if latency_max and latency_min > 0 and cost == "latency":
                        if pruned_cost > latency_max or pruned_cost < latency_min:
                            satisfied_constraint = False
                            break
    
                    if param_max > 0 and param_min > 0 and cost == "param":
                        if pruned_cost > param_max or pruned_cost < param_min:
                            satisfied_constraint = False
                            break

                if satisfied_constraint:
                    sampling_feature = feature
                    break

                try_count += 1

            return sampling_feature
    
    def is_satisfied_constraint(self, feature):
        sampling = torch.Tensor()
        active = torch.Tensor()
        for node_name in self.traversal_order:
            cur_node = self.net.node[node_name]
            
            sampling, active = \
                self.path_recorder.add_sampling(node_name,
                                                torch.as_tensor([feature[cur_node['sampling_param']]]).reshape(
                                                    [1, 1, 1, 1]),
                                                sampling,
                                                active,
                                                self.blocks[cur_node['module']].structure_fixed)

        satisfied_constraint = True
        for cost, cost_eval in self.arch_cost_evaluators.items():
            sampled_arc, pruned_arc = \
                self.path_recorder.get_arch(self.out_node, sampling, active)
            sampled_cost, pruned_cost = \
                cost_eval.get_costs([sampled_arc, pruned_arc])
    
            if self.arch_objective_comp_max > 0 and self.arch_objective_comp_min > 0 and cost == "comp":
                if pruned_cost > self.arch_objective_comp_max or pruned_cost < self.arch_objective_comp_min:
                    satisfied_constraint = False
                    break
    
            if self.arch_objective_latency_max and self.arch_objective_latency_min > 0 and cost == "latency":
                if pruned_cost > self.arch_objective_latency_max or pruned_cost < self.arch_objective_latency_min:
                    satisfied_constraint = False
                    break
    
            if self.arch_objective_param_max > 0 and self.arch_objective_param_min > 0 and cost == "param":
                if pruned_cost > self.arch_objective_param_max or pruned_cost < self.arch_objective_param_min:
                    satisfied_constraint = False
                    break
                    
        return satisfied_constraint
    
    def search_and_plot(self, path=None):
        if not os.path.exists(path):
            os.makedirs(path)

        pass

    def __str__(self):
        model_descr = 'Model:{}\n\t{} nodes\n\t{} blocks\n\t{} parametrized layers\n\t{} computation steps\n\t{} parameters\n\t{} meta-params'
        return model_descr.format(type(self).__name__,
                                  self.graph.number_of_nodes(),
                                  len(self.blocks),
                                  self.n_layers,
                                  self.n_comp_steps,
                                  sum(i.numel() for i in self.parameters()),
                                  len(self.sampling_parameters))