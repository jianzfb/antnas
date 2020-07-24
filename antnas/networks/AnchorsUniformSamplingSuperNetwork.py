# -*- coding: UTF-8 -*-
# @Time    : 2020-04-07 17:57
# @File    : AnchorsUniformSamplingSuperNetwork.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antnas.networks.UniformSamplingSuperNetwork import *
import numpy as np


class AnchorsUniformSamplingSuperNetwork(UniformSamplingSuperNetwork):
    def __init__(self, *args, **kwargs):
        super(AnchorsUniformSamplingSuperNetwork, self).__init__(*args, **kwargs)
        self.anchor_arch_prob = 0.1

    def init(self, *args, **kwargs):
        super(AnchorsUniformSamplingSuperNetwork, self).init(*args, **kwargs)

    def forward(self, x, y, arc=None, epoch=None, warmup=False):
        # 1.step parse x,y - (data,label)
        input = [x]

        # 2.step get sampling architecture of
        anchor_num = self.anchors.size()
        batch_size = input[0].size(0)
        batched_sampling = None
        anchor_arc_pos = None
        is_training_anchor_arc = False
        if np.random.random() < self.anchor_arch_prob and self.training:
            # using anchor arch
            anchor_arc_index = np.random.randint(0, self.anchors.size())
            anchor_arc = self.anchors.arch(anchor_arc_index)
            batched_sampling = torch.as_tensor([anchor_arc], device=x.device)
    
            anchor_arc_pos = torch.ones((batch_size), dtype=torch.int32, device=x.device) * anchor_arc_index
            is_training_anchor_arc = True
        else:
            batched_sampling = arc[0, :].view((1, arc.shape[1]))
            anchor_arc_pos = torch.ones((batch_size), dtype=torch.int32, device=x.device) * (-1)
        
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

            if is_training_anchor_arc and (arc is None) and (node.startswith('CELL') or node.startswith('T')):
                node_output_dict[node] = out[:, :, :, :]
            elif (arc is None) and (node.startswith('CELL') or node.startswith('T')):
                node_output_dict[node] = torch.zeros(out.size(), device=out.device)
                
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
