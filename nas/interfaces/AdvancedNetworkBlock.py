# -*- coding: UTF-8 -*-
# @Time    : 2019-09-19 15:18
# @File    : AdvancedNetworkBlock.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import torch.nn.functional as F
from torch import nn
import torch
import json
import os
import threading
from nas.interfaces.NetworkBlock import *


class RegionSEBlock(NetworkBlock):
    n_layers = 0
    n_comp_steps = 0

    def __init__(self, in_channels, squeeze_channel, region_size):
        super(RegionSEBlock, self).__init__()
        self.in_channels = in_channels
        self.squeeze_channel = squeeze_channel
        self.region_size = region_size

        self.pool2d = torch.nn.AvgPool2d(region_size, region_size, padding=region_size//2)
        self.conv_1 = nn.Conv2d(in_channels, squeeze_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_1 = nn.BatchNorm2d(squeeze_channel)

        self.conv_2 = nn.Conv2d(squeeze_channel, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_2 = nn.BatchNorm2d(in_channels)

        self.switch = True

        self.params = {
            'module_list': ['RegionSEBlock'],
            'name_list': ['RegionSEBlock'],
            'RegionSEBlock': {'in_channels': in_channels, 'squeeze_channel': squeeze_channel, 'region_size': region_size},
        }

    def forward(self, x):
        input = x
        pool2d_x = self.pool2d(input)
        x = self.conv_1(pool2d_x)
        x = self.bn_1(x)
        x = F.relu6(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = F.sigmoid(x)

        x = F.upsample(x, size=(int(input.shape[2]),int(input.shape[3])))
        x = input * x
        x = input + x

        if self.get_sampling() is None:
            return x

        return x * (self._sampling.value == 1).float()

    def get_flop_cost(self, x):
        return [0] * self.state_num


class GCN(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, in_chan, out_chan, k_size=3, bias=True, boundary_refinement=True):
        super(GCN, self).__init__()

        self.params = {
            'module_list': ['GCN'],
            'name_list': ['GCN'],
            'GCN': {'out_chan': out_chan,
                    'k_size': k_size,
                    'bias': bias,
                    'boundary_refinement': boundary_refinement}
        }

        self.left_conv1 = nn.Conv2d(in_chan,
                                    out_chan,
                                    kernel_size=[k_size, 1],
                                    stride=1,
                                    padding=[k_size//2, 0],
                                    bias=True)
        self.left_conv2 = nn.Conv2d(out_chan,
                                    out_chan,
                                    kernel_size=[1, k_size],
                                    stride=1,
                                    padding=[0, k_size//2],
                                    bias=True)

        self.right_conv1 = nn.Conv2d(in_chan,
                                     out_chan,
                                     kernel_size=[1, k_size],
                                     stride=1,
                                     padding=[0, k_size//2],
                                     bias=True)
        self.right_conv2 = nn.Conv2d(out_chan,
                                     out_chan,
                                     kernel_size=[k_size, 1],
                                     stride=1,
                                     padding=[k_size//2, 0],
                                     bias=True)
        self.br = BoundaryRefinement(out_chan, out_chan)

        self.in_chan = in_chan
        self.out_chan = out_chan
        self.boundary_refinement = boundary_refinement

    def forward(self, x):
        left_x1 = self.left_conv1(x)
        left_x2 = self.left_conv2(left_x1)
        right_x1 = self.right_conv1(x)
        right_x2 = self.right_conv2(right_x1)
        x = left_x2 + right_x2

        if self.get_sampling() is None:
            return x

        return x * (self._sampling.value == 1).float()

    def get_flop_cost(self, x):
        conv_in_data_size = torch.Size([1, *x.shape[1:]])
        conv_out_data_size = torch.Size([1, self.out_chan, x.shape[-1], x.shape[-1]])

        flops_1 = self.get_conv2d_flops(self.left_conv1, conv_in_data_size, conv_out_data_size)
        flops_2 = self.get_conv2d_flops(self.left_conv2, conv_out_data_size, conv_out_data_size)
        flops_3 = self.get_conv2d_flops(self.right_conv1, conv_in_data_size, conv_out_data_size)
        flops_4 = self.get_conv2d_flops(self.right_conv2, conv_out_data_size, conv_out_data_size)
        flops_5 = conv_out_data_size.numel() / conv_out_data_size[0]

        flops_6 = 0.0
        if self.boundary_refinement:
            flops_6 = self.br.get_flop_cost(torch.zeros(1, self.out_chan, x.shape[2], x.shape[3]))[1]

        flop_cost = flops_1+flops_2+flops_3+flops_4+flops_5+flops_6
        return [0] + [flop_cost] + [0]*(self.state_num - 2)


class BoundaryRefinement(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, in_chan, out_chan):
        super(BoundaryRefinement, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1, bias=True)
        self.params = {
            'module_list': ['BoundaryRefinement'],
            'name_list': ['BoundaryRefinement'],
            'BoundaryRefinement': {
                'out_chan': out_chan}
        }

        self.switch = True

    def forward(self, x):
        x_res = self.conv1(x)
        x_res = F.relu6(x_res)
        x_res = self.conv2(x_res)

        x = x + x_res
        if self.get_sampling() is None:
            return x

        return x * (self._sampling.value == 1).float()

    def get_flop_cost(self, x):
        conv_in_data_size = torch.Size([1, *x.shape[1:]])
        conv_out_data_size = torch.Size([1, self.conv1.out_channels, x.shape[-1], x.shape[-1]])

        flops_1 = self.get_conv2d_flops(self.conv1, conv_in_data_size, conv_out_data_size)
        flops_2 = self.get_relu_flops(None, conv_out_data_size,conv_out_data_size)
        flops_3 = self.get_conv2d_flops(self.conv2, conv_out_data_size,conv_out_data_size)

        flops = flops_1+flops_2+flops_3
        return [0] + [flops] + [0]*(NetworkBlock.state_num-2)


class ASPPBlock(NetworkBlock):
    n_layers = 0
    n_comp_steps = 0

    def __init__(self, in_chan, depth, atrous_rates):
        super(ASPPBlock, self).__init__()
        self.atrous_rates = atrous_rates
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        # 1.step
        self.conv_1_step = nn.Conv2d(in_chan, depth, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(depth)

        # 2.step
        self.conv_2_step = nn.Conv2d(in_chan, depth, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(depth)

        # 3.step
        self.atrous_conv_list = nn.ModuleList([])
        for i, rate in enumerate(self.atrous_rates):
            self.atrous_conv_list.append(SepConvBN(in_chan, depth, relu=True, k_size=3, dilation=rate))

        # 5.step
        self.conv_5_step = nn.Conv2d((len(self.atrous_rates)+2)*depth, depth, kernel_size=1)
        self.bn5 = nn.BatchNorm2d(depth)

        self.params = {
            'module_list': ['ASPPBlock'],
            'name_list': ['ASPPBlock'],
            'ASPPBlock': {'in_chan': in_chan,
                          'depth': depth,
                          'atrous_rates': atrous_rates},
        }
        self.depth = depth

    def forward(self, x):
        h = x.shape[2]
        w = x.shape[3]
        branch_logits = []

        # 1.step global pooling
        feature_1 = self.global_pool(x)
        feature_1 = self.conv_1_step(feature_1)
        feature_1 = self.bn1(feature_1)
        feature_1 = F.relu6(feature_1)
        feature_1 = F.upsample(feature_1, size=[h, w])
        branch_logits.append(feature_1)

        # 2.step 1x1 convolution
        feature_2 = self.conv_2_step(x)
        feature_2 = self.bn2(feature_2)
        feature_2 = F.relu6(feature_2)
        branch_logits.append(feature_2)

        # 3.step 3x3 convolutions with different atrous rates
        for i in range(len(self.atrous_conv_list)):
            f = self.atrous_conv_list[i](x)
            branch_logits.append(f)

        # 4.step concat
        concat_logits = torch.cat(branch_logits, 1)
        concat_logits = self.conv_5_step(concat_logits)
        concat_logits = self.bn5(concat_logits)
        concat_logits = F.relu6(concat_logits)

        if self.get_sampling() is None:
            return concat_logits

        return concat_logits * (self._sampling.value == 1).float()

    def get_flop_cost(self, x):
        flops = self.get_conv2d_flops(self.conv_1_step, torch.Size((1, x.shape[1], 1, 1)), torch.Size((1, self.depth, 1, 1)))
        flops += self.get_bn_flops(self.bn1, torch.Size((1, self.depth, 1, 1)), torch.Size((1, self.depth, 1, 1)))
        flops += self.get_relu_flops(None, torch.Size((1, self.depth, 1, 1)), torch.Size((1, self.depth, 1, 1)))

        flops += self.get_conv2d_flops(self.conv_2_step,
                                       torch.Size((1, x.shape[1], x.shape[2], x.shape[3])),
                                       torch.Size((1, self.depth, x.shape[2], x.shape[3])))
        flops += self.get_bn_flops(self.bn2, torch.Size((1, self.depth, x.shape[2], x.shape[3])), torch.Size((1, self.depth, x.shape[2], x.shape[3])))
        flops += self.get_relu_flops(None, torch.Size((1, self.depth, x.shape[2], x.shape[3])), torch.Size((1, self.depth, x.shape[2], x.shape[3])))

        for i in range(len(self.atrous_conv_list)):
            flops += self.atrous_conv_list[i].get_flop_cost(x)[1]

        flops += self.get_conv2d_flops(self.conv_5_step,
                                       torch.Size((1, (len(self.atrous_rates)+2)*self.depth, x.shape[2], x.shape[3])),
                                       torch.Size((1, self.depth, x.shape[2], x.shape[3])))
        flops += self.get_bn_flops(self.bn5, torch.Size((1, self.depth, x.shape[2], x.shape[3])), torch.Size((1, self.depth, x.shape[2], x.shape[3])))
        flops += self.get_relu_flops(None, torch.Size((1, self.depth, x.shape[2], x.shape[3])), torch.Size((1, self.depth, x.shape[2], x.shape[3])))

        return [0] + [flops] + [0]*(NetworkBlock.state_num-2)


class FocusBlock(NetworkBlock):
    n_layers = 0
    n_comp_steps = 0

    def __init__(self, in_chan, out_chan):
        super(FocusBlock, self).__init__()
        self.sep_conv1 = SepConvBN(in_chan, out_chan, dilation=1, k_size=3, relu=True)
        self.sep_conv2 = SepConvBN(out_chan, out_chan, dilation=2, k_size=3, relu=True)
        self.sep_conv3 = SepConvBN(out_chan, out_chan, dilation=4, k_size=3, relu=True)
        self.in_chan = in_chan
        self.out_chan = out_chan

        self.params = {
            'module_list': ['FocusBlock'],
            'name_list': ['FocusBlock'],
            'FocusBlock': {'in_chan': in_chan, 'out_chan': out_chan},
        }

    def forward(self, x):
        y1 = self.sep_conv1(x)
        y2 = self.sep_conv2(y1)
        y3 = self.sep_conv3(y2)

        res = y1+y2+y3

        if self.get_sampling() is None:
            return res

        return res * (self._sampling.value == 1).float()

    def get_flop_cost(self, x):
        flops = self.sep_conv1.get_flop_cost(x)[1]
        flops += self.sep_conv2.get_flop_cost(x)[1]
        flops += self.sep_conv3.get_flop_cost(x)[1]

        flops += x.shape[2]*x.shape[3]*self.out_chan*2
        return [0] + [flops] + [0] * (NetworkBlock.state_num - 2)