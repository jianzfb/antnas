# -*- coding: UTF-8 -*-
# @Time    : 2019-08-19 18:16
# @File    : NetworkCell.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from nas.interfaces.NetworkBlock import *
import torch.nn.functional as F


class CellBlock(NetworkBlock):
    n_layers = 3
    n_comp_steps = 1

    def __init__(self, channles, out_channels, reduction=False):
        super(CellBlock, self).__init__()
        self.channles = channles
        self.op_list = nn.ModuleList()
        self.params = {
            'module_list': ['Skip',
                            'InvertedResidualBlockWithSEHS',
                            'InvertedResidualBlockWithSEHS',
                            'InvertedResidualBlockWithSEHS',
                            'InvertedResidualBlockWithSEHS'],
            'name_list': ['Skip',
                          'IRB_k3e3_nohs',
                          'IRB_k3e3_nohs_nose',
                          'IRB_k3e6',
                          'IRB_k3e6_nose'],
            'Skip': {'in_chan': channles,
                     'out_chan': out_channels,
                     'reduction': reduction},
            'IRB_k3e3_nohs': {'in_chan': channles,
                              'kernel_size': 3,
                              'expansion': 3,
                              'out_chan': out_channels,
                              'reduction': reduction,
                              'skip': True,
                              'ratio': 4,
                              'hs': False,
                              'se': True},
            'IRB_k3e3_nohs_nose': {'in_chan': channles,
                                   'kernel_size': 3,
                                   'expansion': 3,
                                   'out_chan': out_channels,
                                   'reduction': reduction,
                                   'skip': True,
                                   'ratio': 4,
                                   'hs': False,
                                   'se': False},
            'IRB_k3e6': {'in_chan': channles,
                         'kernel_size': 3,
                         'expansion': 6,
                         'out_chan': out_channels,
                         'reduction': reduction,
                         'skip': True,
                         'ratio': 4,
                         'hs': True,
                         'se': True},
            'IRB_k3e6_nose': {'in_chan': channles,
                              'kernel_size': 3,
                              'expansion': 6,
                              'out_chan': out_channels,
                              'reduction': reduction,
                              'skip': True,
                              'ratio': 4,
                              'hs': True,
                              'se': False}
        }

        self.op_list = self.build()
        assert(len(self.op_list) == NetworkBlock.state_num)

    def forward(self, input):
        last_cell_result = None
        val_list = []
        for i in range(len(self.op_list)):
            op_result = self.op_list[i](input)

            op_sampling = (self._sampling.value == i).float()
            op_result = op_result * op_sampling

            if getattr(self._last_sampling, 'value', None) is not None:
                if int(self._last_sampling.value.item()) == i:
                    last_cell_result = op_result

            val_list.append(op_result)
        cell_result = sum(val_list)

        # set regularizer loss
        if last_cell_result is not None:
            last_cell_result = last_cell_result.detach()
            regularizer_loss = F.kl_div(cell_result, last_cell_result, reduction='batchmean')
            self.set_node_regularizer(regularizer_loss)

        return cell_result

    def get_flop_cost(self, x):
        cost_list = [0]
        for i in range(len(self.op_list) - 1):
            cost_list.append(self.op_list[i+1].get_flop_cost(x)[1])

        return cost_list

    def get_latency(self, x):
        cost_list = [0]
        for i in range(len(self.op_list) - 1):
            cost_list.append(self.op_list[i + 1].get_latency(x)[1])

        return cost_list


class ReductionCellBlock(NetworkBlock):
    n_layers = 3
    n_comp_steps = 1

    def __init__(self, channles, out_channels):
        super(ReductionCellBlock, self).__init__()
        self.channles = channles
        self.op_list = nn.ModuleList()
        self.params = {
            'module_list': ['InvertedResidualBlockWithSEHS',
                            'InvertedResidualBlockWithSEHS',
                            'InvertedResidualBlockWithSEHS',
                            'InvertedResidualBlockWithSEHS',
                            'InvertedResidualBlockWithSEHS'],
            'name_list': ['IRB_k5e3_nose',
                          'IRB_k3e6_nohs',
                          'IRB_k3e3_nohs_nose',
                          'IRB_k3e6',
                          'IRB_k3e3_nose'],
            'IRB_k5e3_nose': {'in_chan': channles,
                              'kernel_size': 5,
                              'expansion': 3,
                              'out_chan': out_channels,
                              'reduction': True,
                              'skip': False,
                              'ratio': 4,
                              'hs': True,
                              'se': False},
            'IRB_k3e6_nohs': {'in_chan': channles,
                              'kernel_size': 3,
                              'expansion': 6,
                              'out_chan': out_channels,
                              'reduction': True,
                              'skip': False,
                              'ratio': 4,
                              'hs': False,
                              'se': True},
            'IRB_k3e3_nohs_nose': {'in_chan': channles,
                                   'kernel_size': 3,
                                   'expansion': 3,
                                   'out_chan': out_channels,
                                   'reduction': True,
                                   'skip': False,
                                   'ratio': 4,
                                   'hs': False,
                                   'se': False},
            'IRB_k3e6': {'in_chan': channles,
                         'kernel_size': 3,
                         'expansion': 6,
                         'out_chan': out_channels,
                         'reduction': True,
                         'skip': False,
                         'ratio': 4,
                         'hs': True,
                         'se': True},
            'IRB_k3e3_nose': {'in_chan': channles,
                              'kernel_size': 3,
                              'expansion': 3,
                              'out_chan': out_channels,
                              'reduction': True,
                              'skip': False,
                              'ratio': 4,
                              'hs': True,
                              'se': False}
        }

        self.op_list = self.build()
        assert(len(self.op_list) == NetworkBlock.state_num)

    def forward(self, input):
        last_cell_result = None
        val_list = []
        for i in range(len(self.op_list)):
            op_result = self.op_list[i](input)

            op_sampling = (self._sampling.value == i).float()
            op_result = op_result * op_sampling

            if getattr(self._last_sampling, 'value', None) is not None:
                if int(self._last_sampling.value.item()) == i:
                    last_cell_result = op_result

            val_list.append(op_result)
        cell_result = sum(val_list)

        # set regularizer loss
        if last_cell_result is not None:
            last_cell_result = last_cell_result.detach()
            regularizer_loss = F.kl_div(cell_result, last_cell_result, reduction='batchmean')
            self.set_node_regularizer(regularizer_loss)

        return cell_result

    def get_flop_cost(self, x):
        cost_list = [0]
        for i in range(len(self.op_list) - 1):
            cost_list.append(self.op_list[i+1].get_flop_cost(x)[1])

        return cost_list

    def get_latency(self, x):
        cost_list = [0]
        for i in range(len(self.op_list) - 1):
            cost_list.append(self.op_list[i + 1].get_latency(x)[1])

        return cost_list


