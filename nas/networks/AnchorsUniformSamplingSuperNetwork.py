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

    def forward(self, x, y, arc=None, epoch=None, warmup=False):
        # 1.step parse x,y - (data,label)
        input = [x]

        # 2.step get sampling architecture of
        anchor_arc = None
        anchor_arc_index = -1
        anchor_arc_loss = 0.0
        if arc is None:
            anchor_arc_list = []
            for anchor_index in range(self.anchors.size()):
                anchor_arc_list.append(self.anchors.arch(anchor_index))
            anchor_arc = torch.as_tensor(anchor_arc_list, device=x.device)

            if anchor_arc is None:
                batched_sampling = self._sample_archs_with_constraint(input[0].size(0), x.device)
            else:
                batched_sampling = self._sample_archs_with_constraint(input[0].size(0)-self.anchors.size(), x.device)
                batched_sampling = torch.cat([batched_sampling, anchor_arc], dim=0)
        else:
            # search and find
            batched_sampling = arc

        # 3.step forward network
        # 3.1.step set the input of network graph
        # running_graph.node[self.in_node]['input'] = [*input]
        data_dict = {}
        data_dict[self.in_node] = [*input]

        model_out = None
        anchor_arc_loss = []
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
                if anchor_arc is not None:
                    anchor_arc_node_output = self.anchors.output(anchor_arc_index, node)
                    anchor_num = anchor_arc_node_output.shape[0]
                    supernetwork_node_output = out[-anchor_num:]
                    anchor_arc_loss.append(torch.mean((supernetwork_node_output-anchor_arc_node_output)**2))

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
        if anchor_arc is not None:
            loss = loss + 0.01 * torch.mean(torch.as_tensor(anchor_arc_loss))

        return loss, model_accuracy, None, None