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
        if arc is None:
            anchor_arc_list = []
            for anchor_index in range(anchor_num):
                anchor_arc_list.append(self.anchors.arch(anchor_index))
            anchor_arc = torch.as_tensor(anchor_arc_list, device=x.device)

            if batch_size > anchor_num:
                batched_sampling = self._sample_archs_with_constraint(batch_size-anchor_num, x.device)
                batched_sampling = torch.cat([batched_sampling, anchor_arc], dim=0)
            else:
                batched_sampling = anchor_arc

            anchor_arc_pos = index[-anchor_num:]
        else:
            # search and find
            batched_sampling = arc

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
            node_sampling = self.get_node_sampling(node, batch_size, batched_sampling)

            # 3.2.step execute node op
            out = self.blocks[cur_node['module']](input, node_sampling)

            if node.startswith('CELL') or node.startswith('T'):
                node_output_dict[node] = out[-anchor_num:, :, :, :]

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