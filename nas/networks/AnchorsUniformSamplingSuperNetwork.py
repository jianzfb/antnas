# -*- coding: UTF-8 -*-
# @Time    : 2020-04-07 17:57
# @File    : AnchorsUniformSamplingSuperNetwork.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from nas.networks.UniformSamplingSuperNetwork import *
import numpy as np


class AnchorsUniformSamplingSuperNetwork(UniformSamplingSuperNetwork):
    def __init__(self, *args, **kwargs):
        super(AnchorsUniformSamplingSuperNetwork, self).__init__(*args, **kwargs)
        self.anchor_arch_prob = 0.1

    def init(self, *args, **kwargs):
        super(AnchorsUniformSamplingSuperNetwork, self).init(*args, **kwargs)

    def forward(self, x, y, arc=None, epoch=None, warmup=False, index=None):
        # 1.step parse x,y - (data,label)
        input = [x]

        # 2.step get sampling architecture of
        anchor_num = self.anchors.size()
        batch_size = input[0].size(0)
        batched_sampling = None
        anchor_arc_pos = None
        is_training_anchor_arc = False
        if arc is None:
            if np.random.random() < 1.0:
                # using anchor arch
                anchor_arc_index = np.random.randint(0, self.anchors.size())
                anchor_arc = self.anchors.arch(anchor_arc_index)
                batched_sampling = torch.as_tensor([anchor_arc], device=x.device)

                # anchor_arc_pos = index
                anchor_arc_pos = np.ones((batch_size), dtype=np.int32) * anchor_arc_index
                is_training_anchor_arc = True
            else:
                # random select arch
                batched_sampling = torch.as_tensor([self.sample_arch()], device=x.device)
                
                # anchor_arc_pos = torch.LongTensor([])
                anchor_arc_pos = np.ones((batch_size), dtype=np.int32) * (-1)
        else:
            # search and find
            batched_sampling = arc[0, :].view((1, arc.shape[1]))

        # 3.step forward network
        # 3.1.step set the input of network graph
        # running_graph.node[self.in_node]['input'] = [*input]
        data_dict = {}
        data_dict[self.in_node] = [*input]

        model_out = None
        node_output_dict = {}

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

            if is_training_anchor_arc and (node.startswith('CELL') or node.startswith('T')):
                node_output_dict[node] = out[:, :, :, :]

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
        loss = indiv_loss.mean()
        return loss, model_accuracy, node_output_dict, anchor_arc_pos

    def sample_arch(self):
        batch_arch_list = []
        with torch.no_grad():
            while len(batch_arch_list) < 1:
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

            return batch_arch_list[0]
