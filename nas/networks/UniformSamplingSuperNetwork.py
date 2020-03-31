# -*- coding: UTF-8 -*-
# @Time    : 2019-09-29 11:14
# @File    : EvolutionSuperNetwork.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from nas.networks.SuperNetwork import SuperNetwork
from nas.component.NetworkBlock import *
from nas.component.NetworkCell import *
from nas.component.PathRecorder import PathRecorder
from nas.component.NetworkBlock import *
import copy
import networkx as nx
import threading


class UniformSamplingSuperNetwork(SuperNetwork):
    def __init__(self, *args, **kwargs):
        super(UniformSamplingSuperNetwork, self).__init__(*args, **kwargs)

    def forward(self, x, y, arc=None):
        # 0.step copy network graph
        running_graph = copy.deepcopy(self.net)

        # 1.step parse x,y - (data,label)
        input = [x]

        # 2.step get sampling architecture of
        if arc is None:
            batched_sampling = self._sample_archs_with_constraint(input[0].size(0), x.device)
        else:
            # search and find
            batched_sampling = torch.as_tensor([arc]).expand([input[0].size(0), len(arc)])

        # 3.step forward network
        # 3.1.step set the input of network graph
        running_graph.node[self.in_node]['input'] = [*input]
        model_out = None
        for node in self.traversal_order:
            cur_node = running_graph.node[node]
            input = self.format_input(cur_node['input'])

            if len(input) == 0:
                raise RuntimeError('Node {} has no inputs'.format(node))

            batch_size = input[0].size(0) if type(input) == list else input.size(0)
            node_sampling = self.get_node_sampling(node, batch_size, batched_sampling)
            # notify path recorder to add sampling
            # sampling, active = self.add_sampling(node, node_sampling, sampling, active, self.blocks[cur_node['module']].switch)

            # 3.2.step execute node op
            out = self.blocks[cur_node['module']](input, node_sampling)

            if node == self.out_node:
                model_out = out
                break

            # 3.3.step set successor input
            for succ in running_graph.successors(node):
                if 'input' not in running_graph.node[succ]:
                    running_graph.node[succ]['input'] = []
                running_graph.node[succ]['input'].append(out)

        # 4.step compute model loss
        indiv_loss = self.loss(model_out, y)

        # 5.step compute model accuracy
        model_accuracy = self.accuray(model_out, y)

        # 6.step total loss
        loss = indiv_loss.mean()
        return loss, model_accuracy, None, None

    def search_and_save(self, folder=None, name=None):
        if not os.path.exists(folder):
            os.makedirs(folder)

        path = os.path.join(folder, name)
        # 1.step save supernet model
        torch.save(self.state_dict(), '%s.supernet.model'%path)

        # 2.step search parel front

    def sampling_param_generator(self, node_name):
        if not (node_name.startswith('CELL') or node_name.startswith('T')):
            # 不可学习，处于永远激活状态
            param_value = [0] + [1000000000000000] + [0] * (NetworkBlock.state_num - 2)
            trainable = False
        else:
            param_value = [1.0/NetworkBlock.state_num]*NetworkBlock.state_num
            trainable = False

        return nn.Parameter(torch.Tensor(param_value), requires_grad=trainable)

    @property
    def n_layers(self):
        return sum([mod.n_layers for mod in self.blocks])

    @property
    def n_comp_steps(self):
        return sum([mod.n_comp_steps for mod in self.blocks])

    def get_node_sampling(self, node_name, batch_size, batched_sampling):
        """
        Get a batch of sampling for the given node on the given output.
        Fires a "sampling" event with the node name and the sampling Variable.
        :param node_name: Name of the node to sample
        :param out: Tensor on which the sampling will be applied
        :return: A Variable brodcastable to out's size, with all dimensions equals to one except the first one (batch)
        """

        sampling_dim = [batch_size] + [1] * 3
        node = self.net.node[node_name]
        node_sampling = batched_sampling[:, node['sampling_param']].contiguous().view(sampling_dim)

        return node_sampling

    def _init_archs(self, x):

        pass

    def sample_arch(self):
        feature = [None for _ in range(len(self.traversal_order))]
        for node_name in self.traversal_order:
            cur_node = self.net.node[node_name]
            if not (node_name.startswith('CELL') or node_name.startswith('T')):
                # 不可学习，处于永远激活状态
                feature[cur_node['sampling_param']] = int(1)
            else:
                if self.blocks[cur_node['module']].switch:
                    feature[cur_node['sampling_param']] = int(np.random.randint(0, 2))
                else:
                    feature[cur_node['sampling_param']] = int(np.random.randint(0, NetworkBlock.state_num))

        return feature

    def _sample_archs_with_constraint(self, batch_size, device):
        batch_size = int(batch_size)
        batch_arch_list = []
        with torch.no_grad():
            while len(batch_arch_list) < batch_size:
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

                    if self.arch_objective_comp > 0 and cost == "comp":
                        if pruned_cost > self.arch_objective_comp:
                            satisfied_constraint = False
                            break

                    if self.arch_objective_latency > 0 and cost == "latency":
                        if pruned_cost > self.arch_objective_latency:
                            satisfied_constraint = False
                            break

                    if self.arch_objective_param > 0 and cost == "param":
                        if pruned_cost > self.arch_objective_param:
                            satisfied_constraint = False
                            break

                if satisfied_constraint:
                    batch_arch_list.append(feature)

            batch_arch = torch.as_tensor(batch_arch_list, device=device)
            return batch_arch


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