# -*- coding: UTF-8 -*-
# @Time    : 2020-04-21 08:12
# @File    : LoadArc.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antnas.networks.SuperNetwork import *
from antnas.utils.drawers.NASDrawer import NASDrawer
import torch
import torch.nn as nn
from antnas.searchspace.Arc import *


class LoadArc(Arc):
    def __init__(self, architecture, graph=None):
        self.graph = nx.read_gpickle(architecture)
        super(LoadArc, self).__init__(self.graph)

    def generate(self, head=None, tail=None):
        assert(tail is not None)
        self.traversal_order = list(nx.topological_sort(self.graph))
        self.blocks = nn.ModuleList([None for _ in range(len(self.traversal_order))])

        for node_index, node_name in enumerate(self.traversal_order):
            if node_index == 0:
                self.in_node = node_name
            if node_index == len(self.traversal_order) - 1:
                self.out_node = node_name

            cur_node = self.graph.node[node_name]

            module_list = cur_node['module_params']['module_list']
            sampled_module_index = cur_node['sampled']
            sampled_module = None
            sampled_module_name = ''
            if len(module_list) == 1:
                # 单模块，开关状态
                if sampled_module_index == 1:
                    if node_index == len(self.traversal_order) - 1:
                        sampled_module = tail
                    else:
                        sampled_module = globals()[module_list[0]]

                    sampled_module_name = cur_node['module_params']['name_list'][0]
                    self.blocks[cur_node['module']] = \
                        sampled_module(**cur_node['module_params'][sampled_module_name])
                else:
                    sampled_module_name = cur_node['module_params']['name_list'][0]
                    self.blocks[cur_node['module']] = Zero(**cur_node['module_params'][sampled_module_name])
                    sampled_module_name = 'ZERO'
            else:
                # 多模块，多选自选择
                sampled_module = globals()[module_list[sampled_module_index]]
                sampled_module_name = cur_node['module_params']['name_list'][sampled_module_index]
                self.blocks[cur_node['module']] = \
                    sampled_module(**cur_node['module_params'][sampled_module_name])

            print('build node %s with sampling %s op' % (node_name, sampled_module_name))

        return self.in_node, self.out_node


if __name__ == '__main__':
    class OutLayer(NetworkBlock):
        n_layers = 1
        n_comp_steps = 1

        def __init__(self, out_shape, in_chan=160, bias=True):
            super(OutLayer, self).__init__()
            self.conv = nn.Conv2d(in_chan, out_shape[0], kernel_size=1, stride=1, padding=0, bias=bias)
            self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

            self.out_shape = out_shape
            self.params = {
                'module_list': ['OutLayer'],
                'name_list': ['OutLayer'],
                'OutLayer': {'out_shape': out_shape, 'in_chan': in_chan},
                'out': 'outname'
            }

        def forward(self, x, sampling=None):
            x = self.conv(x)
            x = self.global_pool(x)
            return x.view(-1, *self.out_shape)

        def get_flop_cost(self, x):
            return [0] + [0] * (self.state_num - 1)

    pk = LoadArc('/Users/jian/Downloads/aa/accuray_0.7268_para_466966272_params_1325376.architecture')
    pk.generate(tail=OutLayer)
    sampled_loss, pruned_loss = pk.arc_loss([1, 3, 32, 32], 'param')
    print(pruned_loss)