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
            'module_list': ['skip', 'IRB_k3e3_skip', 'IRB_k5e3_skip', 'IRB_k3e6_skip', 'IRB_k5e6_skip'],
            'skip': {'out_chan': out_channels, 'reduction': reduction},
            'IRB_k3e3_skip': {'kernel_size': 3,
                              'expansion': 3,
                              'out_chan': out_channels,
                              'reduction': reduction,
                              'skip': True,
                              'ratio': 4},
            'IRB_k5e3_skip': {'kernel_size': 5,
                              'expansion': 3,
                              'out_chan': out_channels,
                              'reduction': reduction,
                              'skip': True,
                              'ratio': 4},
            'IRB_k3e6_skip': {'kernel_size': 3,
                              'expansion': 6,
                              'out_chan': out_channels,
                              'reduction': reduction,
                              'skip': True,
                              'ratio': 4},
            'IRB_k5e6_skip': {'kernel_size': 5,
                              'expansion': 6,
                              'out_chan': out_channels,
                              'reduction': reduction,
                              'skip': True,
                              'ratio': 4}
        }

        # zero state
        self.skip = Skip(channles, out_channels, reduction=reduction)
        self.op_list.append(self.skip)

        # k3e3 with skip InvertedResidualBlock,
        self.IRB_k3e3_skip = InvertedResidualBlockWithSE(in_chan=channles,
                                                   expansion=3,
                                                   kernel_size=3,
                                                   out_chan=out_channels,
                                                   skip=True,
                                                   reduction=reduction)
        self.op_list.append(self.IRB_k3e3_skip)

        # k5e3 with skip InvertedResidualBlock
        self.IRB_k5e3_skip = InvertedResidualBlockWithSE(in_chan=channles,
                                                   expansion=3,
                                                   kernel_size=5,
                                                   out_chan=out_channels,
                                                   skip=True,
                                                   reduction=reduction)
        self.op_list.append(self.IRB_k5e3_skip)

        # k3e6 with skip InvertedResidualBlock
        self.IRB_k3e6_skip = InvertedResidualBlockWithSE(in_chan=channles,
                                                  expansion=6,
                                                  kernel_size=3,
                                                  out_chan=out_channels,
                                                  skip=True,
                                                  reduction=reduction)
        self.op_list.append(self.IRB_k3e6_skip)

        # k5e6 with skip InvertedResidualBlock
        self.IRB_k5e6_skip = InvertedResidualBlockWithSE(in_chan=channles,
                                                  expansion=6,
                                                  kernel_size=5,
                                                  out_chan=out_channels,
                                                  skip=True,
                                                  reduction=reduction)
        self.op_list.append(self.IRB_k5e6_skip)

        # self.IRB_k5e6_no_skip = InvertedResidualBlock(in_chan=channles,
        #                                              expansion=6,
        #                                              kernel_size=5,
        #                                              out_chan=channles,
        #                                              skip=False)
        # self.op_list.append(self.IRB_k5e6_no_skip)
        #
        # # k3 with skip SepConv
        # self.SC_k3_skip = SepConv(in_chan=channles, kernel_size=3, skip=True)
        # self.op_list.append(self.SC_k3_skip)
        #
        # self.SC_k3_no_skip = SepConv(in_chan=channles, kernel_size=3, skip=False)
        # self.op_list.append(self.SC_k3_no_skip)
        #
        # # k5 with skip SepConv
        # self.SC_k5_skip = SepConv(in_chan=channles, kernel_size=5, skip=True)
        # self.op_list.append(self.SC_k5_skip)
        #
        # self.SC_k5_no_skip = SepConv(in_chan=channles, kernel_size=5, skip=False)
        # self.op_list.append(self.SC_k5_no_skip)

    def forward(self, input):
        if self._sampling is None:
            return self.op_list[-1](input)

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
