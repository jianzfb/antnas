# -*- coding: UTF-8 -*-
# @Time    : 2019-08-19 18:16
# @File    : NetworkBlock.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import torch.nn.functional as F
from torch import nn
import numpy as np
import torch
import json
import os


class NetworkBlock(nn.Module):
    state_num = 5
    device_num = 1  # 1 or 2
    bn_moving_momentum = False
    bn_track_running_stats = False
    lookup_table = {}

    def __init__(self):
        super(NetworkBlock, self).__init__()
        self._params = {}
        self._structure_fixed = True

    @property
    def structure_fixed(self):
        return self._structure_fixed

    @structure_fixed.setter
    def structure_fixed(self, val):
        self._structure_fixed = val

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, val):
        self._params = val

    def build(self, *args, **kwargs):
        op_list = nn.ModuleList()
        module_list = self.params['module_list']
        name_list = self.params['name_list']
        for block_name, block_module in zip(name_list, module_list):
            block_param = self.params[block_name]
            op_list.append(globals()[block_module](**block_param))

        return op_list

    def get_latency(self, x):
        if NetworkBlock.device_num == 1:
            return [0.01] * NetworkBlock.state_num
        else:
            return [[0.01] * NetworkBlock.state_num, [0.01] * NetworkBlock.state_num]

    def get_param_num(self, x):
        return [0.0] * NetworkBlock.state_num

    def get_flop_cost(self, x):
        return [0.0] * NetworkBlock.state_num

    @staticmethod
    def load_lookup_table(lookup_table_file):
        if os.path.exists(lookup_table_file):
            with open(lookup_table_file) as fp:
                NetworkBlock.lookup_table = json.load(fp)
            return True
        else:
            return False

    @staticmethod
    def proximate_latency(op_name, profile, device='cpu', mode='soft'):
        if op_name in NetworkBlock.lookup_table[device]['op']:
            if profile in NetworkBlock.lookup_table[device]['op'][op_name]['latency']:
                return NetworkBlock.lookup_table[device]['op'][op_name]['latency'][profile]
            else:
                if mode == 'soft':
                    # 采用近似方式获取
                    input_s,input_c,output_s,output_c = profile.split('x')
                    soft_profile = '%dx%dx%dx%d'%((int)(output_s), (int)(input_c),(int)(output_s),(int)(output_c))
                    if soft_profile in NetworkBlock.lookup_table[device]['op'][op_name]['latency']:
                        return NetworkBlock.lookup_table[device]['op'][op_name]['latency'][soft_profile]

        print('(%s - %s) not in latency lookup table'%(op_name, profile))
        return 0.01

    @staticmethod
    def get_conv2d_flops(m, x_size=None, y_size=None):
        cin = m.in_channels
        cout = m.out_channels
        kh, kw = m.kernel_size

        out_h = y_size[2]
        out_w = y_size[3]

        kernel_multi_ops = kh * kw * cin // m.groups
        kernel_add_ops = kh * kw * cin // m.groups - 1
        bias_ops = 1 if m.bias is not None else 0
        ops_per_element = kernel_multi_ops + kernel_add_ops + bias_ops

        # total ops
        output_elements = out_w * out_h * cout
        total_ops = output_elements * ops_per_element

        return total_ops

    @staticmethod
    def get_bn_flops(m=None, x_size=None, y_size=None):
        nelements = 1 * x_size[1] * x_size[2] * x_size[3]
        total_ops = 4 * nelements
        return total_ops

    @staticmethod
    def get_relu_flops(m=None, x_size=None, y_size=None):
        # nelements = 1 * x_size[1] * x_size[2] * x_size[3]
        nelements = x_size.numel() / x_size[0]
        total_ops = nelements
        return total_ops

    @staticmethod
    def get_hs_flops(m, x_size, y_size=None):
        nelements = x_size.numel() / x_size[0]
        total_ops = nelements * 4
        return total_ops

    @staticmethod
    def get_avgpool_flops(m, x_size, y_size):
        total_add = m.kernel_size[0] * m.kernel_size[1]
        total_div = 1

        kernel_ops = total_add + total_div
        # num_elements = 1 * y_size[1] * y_size[2] * y_size[3]
        num_elements = y_size.numel() / y_size[0]
        total_ops = kernel_ops * num_elements
        return total_ops

    @staticmethod
    def get_avgglobalpool_flops(m, x_size, y_size):
        total_add = (x_size[2] // m.output_size[0]) * (x_size[3] // m.output_size[1])
        total_div = 1

        kernel_ops = total_add + total_div
        # num_elements = 1 * y_size[1] * y_size[2] * y_size[3]
        num_elements = y_size.numel() / y_size[0]
        total_ops = kernel_ops * num_elements
        return total_ops

    @staticmethod
    def get_linear_flops(m, x_size, y_size):
        total_mul = m.in_features
        total_add = m.in_features - 1
        num_elements = y_size.numel() / y_size[0]
        total_ops = (total_mul + total_add) * num_elements

        return total_ops


class Identity(NetworkBlock):
    n_layers = 0
    n_comp_steps = 0

    def __init__(self, in_chan=0, out_chan=0):
        super(Identity, self).__init__()
        self.structure_fixed = False

        self.params = {
            'module_list': ['Identity'],
            'name_list': ['Identity'],
            'Identity': {},
            'in_chan': in_chan,
            'out_chan': out_chan
        }

    def forward(self, x, sampling=None):
        if sampling is None:
            return x

        is_activate = (int)(sampling.item())
        if is_activate == 1:
            return x
        else:
            return torch.zeros(x.shape, device=x.device)

    def get_flop_cost(self, x):
        return [0] * self.state_num


class Zero(NetworkBlock):
    n_layers = 0
    n_comp_steps = 0

    def __init__(self, in_chan, out_chan, reduction=False, **kwargs):
        super(Zero, self).__init__()
        self.structure_fixed = True
        self.in_channels = in_chan
        self.out_channels = out_chan
        self.reduction = reduction
        self.params = {
            'module_list': ['Zero'],
            'name_list': ['Zero'],
            'Zero': {'out_chan': out_chan, 'reduction': reduction, 'in_chan': in_chan},
            'in_chan': in_chan,
            'out_chan': out_chan
        }

    def forward(self, x, sampling=None):
        batch_size = x.size(0)
        channels = x.size(1)
        H = x.size(2)
        W = x.size(3)
        if self.reduction:
            H = H//2
            W = W//2
        
        if self.out_channels != self.in_channels:
            return torch.zeros([batch_size, self.out_channels, H, W], device=x.device)
        else:
            return torch.zeros([batch_size, channels, H, W], device=x.device)

    def get_flop_cost(self, x):
        return [0] * self.state_num

    def get_param_num(self, x):
        return [0] * self.state_num

    def get_latency(self, x):
        if NetworkBlock.device_num == 1:
            return [0.01] * self.state_num
        else:
            return [[0.01] * self.state_num, [0.01] * self.state_num]


class Skip(NetworkBlock):
    n_layers = 0
    n_comp_steps = 0

    def __init__(self, in_chan, out_chan, reduction=False):
        super(Skip, self).__init__()
        self.in_channels = in_chan
        self.out_channels = out_chan
        self.reduction = reduction
        self.pool2d = None
        if reduction:
            self.pool2d = torch.nn.AvgPool2d(2, 2)
        self.structure_fixed = False

        self.params = {
            'module_list': ['Skip'],
            'name_list': ['Skip'],
            'Skip': {'out_chan': out_chan, 'reduction': reduction, 'in_chan': in_chan},
            'in_chan': in_chan,
            'out_chan': out_chan
        }

    def forward(self, x, sampling=None):
        x_res = x
        if self.out_channels > self.in_channels:
            x_res = torch.cat([x, torch.zeros(x.size(0),
                                              (self.out_channels-self.in_channels),
                                              x.size(2),
                                              x.size(3), device=x.device)], dim=1)
        elif self.out_channels < self.in_channels:
            x_res = x[:, 0:self.out_channels, :, :]
        
        if self.reduction:
            x_res = self.pool2d(x_res)

        if sampling is None:
            return x_res

        is_activate = (int)(sampling.item())
        if is_activate == 1:
            return x_res
        else:
            return torch.zeros(x_res.shape, device=x.device)

    def get_flop_cost(self, x):
        return [0] * NetworkBlock.state_num

    def get_latency(self, x):
        if NetworkBlock.device_num == 1:
            return [0.01] * NetworkBlock.state_num
        else:
            return [[0.01] * self.state_num, [0.01] * self.state_num]


class ConvBn(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, in_chan, out_chan, relu, k_size=3, stride=1, dilation=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=k_size, stride=stride, padding=k_size//2, bias=False, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_chan,
                                 momentum=1.0 if not NetworkBlock.bn_moving_momentum else 0.1,
                                 track_running_stats=NetworkBlock.bn_track_running_stats)
        self.relu = relu
        self.out_chan = out_chan
        self.params = {
            'module_list': ['ConvBn'],
            'name_list': ['ConvBn'],
            'ConvBn': {'stride': stride,
                       'out_chan': out_chan,
                       'k_size': k_size,
                       'relu': relu,
                       'dilation': dilation,
                       'in_chan': in_chan},
            'in_chan': in_chan,
            'out_chan': out_chan
        }
        self.structure_fixed = False

    def get_param_num(self, x):
        return [0] + \
               [self.conv.kernel_size[0]*self.conv.kernel_size[1]*self.conv.in_channels*self.conv.out_channels] + \
               [0]*(NetworkBlock.state_num - 2)

    def forward(self, x, sampling=None):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = F.relu(x)
        if sampling is None:
            return x

        is_activate = (int)(sampling.item())
        if is_activate == 1:
            return x
        else:
            return torch.zeros(x.shape, device=x.device)

    def get_flop_cost(self, x):
        conv_in_data_size = torch.Size([1, *x.shape[1:]])
        conv_out_data_size = torch.Size([1, self.out_chan, x.shape[-1]//self.conv.stride[0], x.shape[-1]//self.conv.stride[1]])

        flops_1 = self.get_conv2d_flops(self.conv, conv_in_data_size, conv_out_data_size)
        flops_2 = self.get_bn_flops(self.bn, conv_out_data_size, conv_out_data_size)
        flops_3 = 0
        if self.relu:
            flops_3 = self.get_relu_flops(None, conv_out_data_size, conv_out_data_size)

        total_flops = flops_1 + flops_2 + flops_3
        flop_cost = [0] + [total_flops] + [0] * (self.state_num - 2)
        return flop_cost

    def get_latency(self, x):
        op_name = "convbn_%dx%d" % (self.conv.kernel_size[0], self.conv.kernel_size[1])

        input_h, _ = x.shape[2:]
        after_h = input_h // self.conv.stride[0]
        op_latency = \
            NetworkBlock.proximate_latency(op_name,
                                           '%dx%dx%dx%d' % (int(input_h),self.conv.in_channels,int(after_h), self.conv.out_channels),
                                           'cpu')

        latency_cost = [0.01] + [op_latency] + [0.01] * (NetworkBlock.state_num - 2)

        if NetworkBlock.device_num > 1:
            op_latency =\
                NetworkBlock.proximate_latency(op_name,
                                               '%dx%dx%dx%d' % (int(input_h),self.conv.in_channels,int(after_h),self.conv.out_channels),
                                               'gpu')
            latency_cost_gpu = [0.01] + [op_latency] + [0.01] * (NetworkBlock.state_num - 2)
            return [latency_cost, latency_cost_gpu]

        return latency_cost


class SepConvBN(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, in_chan, out_chan, relu, k_size=3, stride=1,  dilation=1):
        super(SepConvBN, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_chan,
                                        in_chan,
                                        kernel_size=k_size,
                                        groups=in_chan,
                                        stride=stride,
                                        padding=k_size // 2 + (dilation-1)*(k_size-1)//2,
                                        dilation=dilation,
                                        bias=False)
        self.depthwise_bn = nn.BatchNorm2d(in_chan,
                                           momentum=1.0 if not NetworkBlock.bn_moving_momentum else 0.1,
                                           track_running_stats=NetworkBlock.bn_track_running_stats)

        self.pointwise_conv = nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_chan,
                                 momentum=1.0 if not NetworkBlock.bn_moving_momentum else 0.1,
                                 track_running_stats=NetworkBlock.bn_track_running_stats)
        self.relu = relu
        self.out_chan = out_chan

        self.params = {
            'module_list': ['SepConvBN'],
            'name_list': ['SepConvBN'],
            'SepConvBN': {'stride': stride,
                          'out_chan': out_chan,
                          'k_size': k_size,
                          'relu': relu,
                          'dilation': dilation,
                          'in_chan': in_chan},
            'in_chan': in_chan,
            'out_chan': out_chan
        }
        self.structure_fixed = False

    def forward(self, x, sampling=None):
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = F.relu(x)

        x = self.pointwise_conv(x)
        x = self.bn(x)
        if self.relu:
            x = F.relu(x)

        if sampling is None:
            return x

        is_activate = int(sampling.item())
        if is_activate == 1:
            return x
        else:
            return torch.zeros(x.shape, device=x.device)

    def get_param_num(self, x):
        part1_params = self.depthwise_conv.kernel_size[0]*self.depthwise_conv.kernel_size[1]*self.depthwise_conv.in_channels
        part2_params = self.pointwise_conv.in_channels * self.pointwise_conv.out_channels
        return [0] + \
               [part1_params+part2_params] + \
               [0]*(NetworkBlock.state_num - 2)

    def get_flop_cost(self, x):
        depthwise_out_data_size = torch.Size([1,
                                              self.depthwise_conv.out_channels,
                                              x.shape[-1] // self.depthwise_conv.stride[0],
                                              x.shape[-1] // self.depthwise_conv.stride[1]])
        pointwise_out_data_size = torch.Size([1,
                                              self.pointwise_conv.out_channels,
                                              x.shape[-1] // self.depthwise_conv.stride[0],
                                              x.shape[-1] // self.depthwise_conv.stride[1]])

        flops_1 = self.get_conv2d_flops(m=self.depthwise_conv, y_size=depthwise_out_data_size)
        flops_2 = self.get_bn_flops(x_size=depthwise_out_data_size)
        flops_3 = self.get_relu_flops(depthwise_out_data_size)

        flops_4 = self.get_conv2d_flops(m=self.pointwise_conv, y_size=pointwise_out_data_size)
        flops_5 = self.get_bn_flops(x_size=pointwise_out_data_size)
        flops_6 = 0
        if self.relu:
            flops_6 = self.get_relu_flops(x_size=pointwise_out_data_size)

        total_flops = flops_1 + flops_2 + flops_3 + flops_4 + flops_5 + flops_6
        flop_cost = [0] + [total_flops] + [0] * (self.state_num - 2)
        return flop_cost

    def get_latency(self, x):
        op_name = "sepconvbn_%dx%d" % (self.depthwise_conv.kernel_size[0], self.depthwise_conv.kernel_size[1])

        input_h, _ = x.shape[2:]
        after_h = input_h // self.depthwise_conv.stride[0]
        op_latency = \
            NetworkBlock.proximate_latency(op_name,
                                           '%dx%dx%dx%d' % (int(input_h),self.depthwise_conv.in_channels,int(after_h),self.depthwise_conv.out_channels),
                                           'cpu')
        latency_cost = [0.01] + [op_latency] + [0.01] * (NetworkBlock.state_num - 2)

        if NetworkBlock.device_num > 1:
            op_latency = \
                NetworkBlock.proximate_latency(op_name,
                                               '%dx%dx%dx%d' % (int(input_h),self.depthwise_conv.in_channels,int(after_h),self.depthwise_conv.out_channels),
                                               'gpu')
            latency_cost_gpu = [0.01] + [op_latency] + [0.01] * (NetworkBlock.state_num - 2)
            return [latency_cost, latency_cost_gpu]

        return latency_cost


class ResizedBlock(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, in_chan, out_chan, relu=True, k_size=3, scale_factor=2):
        super(ResizedBlock, self).__init__()
        if out_chan > 0:
            self.conv_layer = ConvBn(in_chan, out_chan, relu=relu, k_size=k_size)
        else:
            self.conv_layer = None

        self.scale_factor = scale_factor

        self.params = {
            'module_list': ['ResizedBlock'],
            'name_list': ['ResizedBlock'],
            'ResizedBlock': {'out_chan': out_chan,
                             'relu': relu,
                             'k_size': k_size,
                             'scale_factor': scale_factor,
                             'in_chan': in_chan},
            'in_chan': in_chan,
            'out_chan': out_chan
        }
        self.structure_fixed = False

    def forward(self, x, sampling=None):
        # x = F.upsample(x, scale_factor=self.scale_factor, mode='bilinear')
        x = torch.nn.functional.interpolate(x,scale_factor=self.scale_factor,mode='bilinear',align_corners=True)
        if self.conv_layer is not None:
            x = self.conv_layer(x)

        if sampling is None:
            return x

        is_activate = (int)(sampling.item())
        if is_activate == 1:
            return x
        else:
            return torch.zeros(x.shape, device=x.device)

    def get_param_num(self, x):
        if self.conv_layer is not None:
            return [0] + [self.conv_layer.get_param_num(x)[1]] + [0] * (NetworkBlock.state_num - 2)
        else:
            return [0] * NetworkBlock.state_num

    def get_flop_cost(self, x):
        flops = 9 * (x.shape[2]*self.scale_factor)*(x.shape[3]*self.scale_factor) * x.shape[1]
        if self.conv_layer is not None:
            # x = F.upsample(x, scale_factor=self.scale_factor, mode='bilinear')
            x = torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)

            flops += self.conv_layer.get_flop_cost(x)[1]
        return [0] + [flops] + [0] * (NetworkBlock.state_num - 2)

    def get_latency(self, x):
        op_name = "resize"

        input_h, _ = x.shape[2:]
        after_h = (int)(self.scale_factor * input_h)
        op_latency = \
            NetworkBlock.proximate_latency(op_name,
                                           '%dx%d' % (int(input_h), int(after_h)),
                                           'cpu')
        latency_cost = [0.01] + [op_latency] + [0.01] * (NetworkBlock.state_num - 2)

        if NetworkBlock.device_num > 1:
            op_latency = \
                NetworkBlock.proximate_latency(op_name,
                                               '%dx%d' % (int(input_h), int(after_h)),
                                               'gpu')
            latency_cost_gpu = [0.01] + [op_latency] + [0.01] * (NetworkBlock.state_num - 2)
            return [latency_cost, latency_cost_gpu]

        return latency_cost


class AddBlock(NetworkBlock):
    n_layers = 0
    n_comp_steps = 1

    def __init__(self):
        super(AddBlock, self).__init__()
        self.params = {
            'module_list': ['AddBlock'],
            'name_list': ['AddBlock'],
            'AddBlock': {},
            'in_chan': 0,
            'out_chan': 0
        }
        self.structure_fixed = True

    def forward(self, x, sampling=None):
        if not isinstance(x, list):
            return x

        assert isinstance(x, list)
        return sum(x)

    def get_flop_cost(self, x):
        if not isinstance(x, list):
            return [0] * self.state_num

        flop_cost = [0] + [x[0].size().numel()/x[0].size()[0] * (len(x) - 1)] * (self.state_num - 1)
        return flop_cost


class ConcatBlock(NetworkBlock):
    n_layers = 0
    n_comp_steps = 1

    def __init__(self):
        super(ConcatBlock, self).__init__()
        self.params = {
            'module_list': ['ConcatBlock'],
            'name_list': ['ConcatBlock'],
            'ConcatBlock': {},
            'in_chan': 0,
            'out_chan': 0
        }
        self.structure_fixed = True

    def forward(self, x, sampling=None):
        if not isinstance(x, list):
            return x

        assert isinstance(x, list)
        return torch.cat(x, dim=1)

    def get_flop_cost(self, x):
        if not isinstance(x, list):
            return [0] * self.state_num

        # just use AddBlock flops (not precise)
        flop_cost = [0] + [x[0].size().numel()/x[0].size()[0] * (len(x) - 1)] * (self.state_num - 1)
        return flop_cost


class MergeBlock(NetworkBlock):
    n_layers = 0
    n_comp_steps = 1

    def __init__(self, in_chan, out_chan):
        super(MergeBlock, self).__init__()
        self.params = {
            'module_list': ['MergeBlock'],
            'name_list': ['MergeBlock'],
            'MergeBlock': {
                'in_chan': in_chan,
                'out_chan': out_chan
            },
            'in_chan': in_chan,
            'out_chan': out_chan
        }
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_chan,
                                 momentum=1.0 if not NetworkBlock.bn_moving_momentum else 0.1,
                                 track_running_stats=NetworkBlock.bn_track_running_stats)
        self.relu = True
        self.structure_fixed = True

    def forward(self, x, sampling=None):
        if not isinstance(x, list):
            return x

        # reduce to same size
        target_size = -1
        tt = []
        for t in x:
            if target_size < 0:
                target_size = t.shape[2]

            target_size = target_size if target_size < t.shape[2] else t.shape[2]

        for t in x:
            if target_size != t.shape[2]:
                ks = t.shape[2] // target_size
                tt.append(torch.nn.AvgPool2d(kernel_size=[ks, ks], padding=(ks-1)//2, stride=ks)(t))
            else:
                tt.append(t)
        tt = torch.cat(tt, dim=1)

        # reduce to channels
        tt = self.conv(tt)
        tt = self.bn(tt)
        tt = F.relu(tt)
        
        return tt

    def get_param_num(self, x):
        conv_param = self.conv.in_channels*self.conv.out_channels*self.conv.kernel_size[0]*self.conv.kernel_size[1]
        return [0]+[conv_param]+[0]*(NetworkBlock.state_num-2)


class MaxPoolingBlock(NetworkBlock):
    n_layers = 0
    n_comp_steps = 1

    def __init__(self, k_size, stride):
        super(MaxPoolingBlock, self).__init__()
        self.params = {
            'module_list': ['MaxPoolingBlock'],
            'name_list': ['MaxPoolingBlock'],
            'MaxPoolingBlock': {
                'k_size': k_size,
                'stride': stride
            },
            'in_chan': 0,
            'out_chan': 0
        }

        self.k_size = k_size
        self.stride = stride
        self.structure_fixed = True

    def forward(self, x, sampling=None):
        x = torch.nn.MaxPool2d(kernel_size=self.k_size,stride=self.stride,padding=(self.k_size-1)//2)(x)
        return x


class AvgPoolingBlock(NetworkBlock):
    n_layers = 0
    n_comp_steps = 1

    def __init__(self, k_size, stride):
        super(AvgPoolingBlock, self).__init__()
        self.params = {
            'module_list': ['AvgPoolingBlock'],
            'name_list': ['AvgPoolingBlock'],
            'AvgPoolingBlock': {
                'k_size': k_size,
                'stride': stride
            },
            'in_chan': 0,
            'out_chan': 0
        }

        self.k_size = k_size
        self.stride = stride
        self.structure_fixed = True

    def forward(self, x, sampling=None):
        x = torch.nn.AvgPool2d(kernel_size=self.k_size,stride=self.stride,padding=(self.k_size-1)//2)(x)
        return x


def _make_divisible(v, divisor=8, min_value=8):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidualBlockWithSEHS(NetworkBlock):
    n_layers = 3
    n_comp_steps = 1

    def __init__(self, in_chan, expansion, kernel_size, out_chan, skip=True,
                 reduction=False, ratio=4, se=True, hs=True,
                 dilation=1):
        super(InvertedResidualBlockWithSEHS, self).__init__()
        self.structure_fixed = False
        self.params = {
            'module_list': ['InvertedResidualBlockWithSEHS'],
            'name_list': ['InvertedResidualBlockWithSEHS'],
            'InvertedResidualBlockWithSEHS': {'in_chan': in_chan,
                                              'expansion': expansion,
                                              'kernel_size': kernel_size,
                                              'out_chan': out_chan,
                                              'skip': skip,
                                              'reduction': reduction,
                                              'ratio': ratio,
                                              'se': se,
                                              'hs': hs,
                                              'dilation': dilation},
            'in_chan': in_chan,
            'out_chan': out_chan
        }

        # expansion,
        expansion_channels = in_chan
        if expansion != 1:
            expansion_channels = _make_divisible(in_chan * expansion)
            self.conv1 = nn.Conv2d(in_chan,
                                   expansion_channels,
                                   kernel_size=1,
                                   stride=1,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(expansion_channels,
                                      momentum=1.0 if not NetworkBlock.bn_moving_momentum else 0.1,
                                      track_running_stats=NetworkBlock.bn_track_running_stats)
        # depthwise
        self.dwconv2 = nn.Conv2d(expansion_channels,
                                 expansion_channels,
                                 kernel_size=kernel_size,
                                 groups=expansion_channels,
                                 stride=2 if reduction else 1,
                                 padding=kernel_size // 2 + (kernel_size-1)*(dilation-1) // 2,
                                 bias=False,
                                 dilation=dilation)
        self.bn2 = nn.BatchNorm2d(expansion_channels,
                                  momentum=1.0 if not NetworkBlock.bn_moving_momentum else 0.1,
                                  track_running_stats=NetworkBlock.bn_track_running_stats)

        # for se
        if se:
            squeeze_channels = _make_divisible(expansion_channels / ratio, divisor=8)
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.se_conv_layer_1 = nn.Conv2d(expansion_channels,
                                             squeeze_channels,
                                             kernel_size=1,
                                             bias=True,
                                             stride=1,
                                             padding=0)
            self.se_conv_layer_2 = nn.Conv2d(squeeze_channels,
                                             expansion_channels,
                                             kernel_size=1,
                                             bias=True,
                                             stride=1,
                                             padding=0)
        # pointwise
        self.conv3 = nn.Conv2d(expansion_channels,
                               out_chan,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_chan,
                                  momentum=1.0 if not NetworkBlock.bn_moving_momentum else 0.1,
                                  track_running_stats=NetworkBlock.bn_track_running_stats)

        self.ratio = ratio
        self.reduction = reduction
        self.kernel_size = kernel_size
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.expansion = expansion
        self.se = se
        self.hs = hs
        self.identity = (not reduction) and (in_chan == out_chan)
        self.skip = skip

    def forward(self, input, sampling=None):
        x = input
        # expansion
        if self.expansion != 1:
            x = self.conv1(x)
            x = self.bn1(x)
            if self.hs:
                x = x * (F.relu6(x + 3.0) / 6.0)
            else:
                x = F.relu(x)
        # depthwise
        x = self.dwconv2(x)
        x = self.bn2(x)
        if self.hs:
            x = x * (F.relu6(x + 3.0) / 6.0)
        else:
            x = F.relu(x)
        # se
        if self.se:
            se_x = self.global_pool(x)
            se_x = self.se_conv_layer_1(se_x)
            se_x = F.relu(se_x)

            se_x = self.se_conv_layer_2(se_x)
            se_x = F.relu6(se_x + 3.0) / 6.0        # hard sigmoid
            x = torch.mul(se_x, x)
        # pointwise
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity and self.skip:
            x = x + input

        if sampling is None:
            return x

        is_activate = int(sampling.item())
        if is_activate == 1:
            return x
        else:
            return torch.zeros(x.shape, device=x.device)

    def get_param_num(self, x):
        conv1_param = 0
        if self.expansion != 1:
            conv1_param = self.conv1.in_channels*self.conv1.out_channels*self.conv1.kernel_size[0]*self.conv1.kernel_size[1]
        
        conv2_param = self.dwconv2.kernel_size[0]*self.dwconv2.kernel_size[1]*self.dwconv2.in_channels
        conv_se_1_param = 0
        conv_se_2_param = 0
        if self.se:
            conv_se_1_param = self.se_conv_layer_1.kernel_size[0]*\
                              self.se_conv_layer_1.kernel_size[1]*\
                              self.se_conv_layer_1.in_channels*\
                              self.se_conv_layer_1.out_channels
            conv_se_2_param = self.se_conv_layer_2.kernel_size[0]*\
                              self.se_conv_layer_2.kernel_size[1]*\
                              self.se_conv_layer_2.in_channels*\
                              self.se_conv_layer_2.out_channels

        conv3_param = self.conv3.kernel_size[0]*self.conv3.kernel_size[1]*self.conv3.in_channels*self.conv3.out_channels

        params = conv1_param+conv2_param+conv_se_1_param+conv_se_2_param+conv3_param
        return [0] + [params] + [0]*(NetworkBlock.state_num - 2)

    def get_flop_cost(self, x):
        step_1_in_size = torch.Size([1, *x.shape[1:]])
        step_1_out_size = [1, self.conv1.out_channels, x.shape[2], x.shape[3]]
        step_1_out_size = torch.Size(step_1_out_size)

        step_2_out_size = [1, self.dwconv2.out_channels, x.shape[2], x.shape[3]]
        if self.reduction:
            step_2_out_size[2] = step_2_out_size[2] // 2
            step_2_out_size[3] = step_2_out_size[3] // 2
        step_2_out_size = torch.Size(step_2_out_size)

        step_3_out_size = torch.Size([1, self.out_chan, step_2_out_size[2], step_2_out_size[3]])

        # expansion flops
        flops_1 = 0.0
        flops_2 = 0.0
        flops_3 = 0.0
        if self.expansion != 1:
            flops_1 = self.get_conv2d_flops(self.conv1, step_1_in_size, step_1_out_size)
            flops_2 = self.get_bn_flops(self.bn1, step_1_out_size, step_1_out_size)
            if self.hs:
                flops_3 = self.get_hs_flops(None, step_1_out_size, step_1_out_size)
            else:
                flops_3 = self.get_relu_flops(F.relu, step_1_out_size, step_1_out_size)

        # depthwise flops
        flops_4 = self.get_conv2d_flops(self.dwconv2, step_1_out_size, step_2_out_size)
        flops_5 = self.get_bn_flops(self.bn2, step_2_out_size, step_2_out_size)
        if self.hs:
            flops_6 = self.get_hs_flops(None, step_2_out_size, step_2_out_size)
        else:
            flops_6 = self.get_relu_flops(F.relu, step_2_out_size, step_2_out_size)

        # se flops
        flops_se = 0.0
        if self.se:
            se_input_size = torch.Size([1, self.se_conv_layer_1.in_channels, 1, 1])
            se_conv1_output_size = torch.Size([1, self.se_conv_layer_1.out_channels, 1, 1])
            se_conv2_output_size = torch.Size([1, self.se_conv_layer_2.out_channels, 1, 1])
            # global_pool
            flops_se_1 = self.get_avgglobalpool_flops(self.global_pool, step_2_out_size, se_input_size)
            # conv1
            flops_se_2 = self.get_conv2d_flops(self.se_conv_layer_1, se_input_size, se_conv1_output_size)
            flops_se_3 = self.get_relu_flops(F.relu, se_conv1_output_size, se_conv1_output_size)
            # conv2
            flops_se_4 = self.get_conv2d_flops(self.se_conv_layer_2, se_conv1_output_size, se_conv2_output_size)
            flops_se_5 = se_conv2_output_size[0] * se_conv2_output_size[1] * \
                         se_conv2_output_size[2] * se_conv2_output_size[3] + \
                         self.get_relu_flops(F.relu6, se_conv2_output_size, se_conv2_output_size)
            # multi
            flops_se_6 = 1 * step_2_out_size[1] * step_2_out_size[2] * step_2_out_size[3]

            flops_se = flops_se_1 + flops_se_2 + flops_se_3 + flops_se_4 + flops_se_5 + flops_se_6

        # pointwise flops
        flops_7 = self.get_conv2d_flops(self.conv3, step_2_out_size, step_3_out_size)
        flops_8 = self.get_bn_flops(self.bn3, step_3_out_size, step_3_out_size)

        total_flops = flops_1 + flops_2 + flops_3 + flops_4 + flops_5 + flops_6 + flops_7 + flops_8 + flops_se

        flop_cost = [0] + [total_flops] * (NetworkBlock.state_num - 2)
        return flop_cost

    def get_latency(self, x):
        irb_name = 'irb_%dx%d'%(self.kernel_size, self.kernel_size)
        if self.se:
            irb_name += "_se"
        if self.hs:
            irb_name += "_hs"

        irb_name += "_e%d"%self.expansion

        # ignore skip flag
        # if not self.reduction:
        #     irb_name += "_skip"

        input_h, _ = x.shape[2:]
        after_h = input_h if not self.reduction else input_h // 2
        irb_latency = \
            NetworkBlock.proximate_latency(irb_name,
                                           "%dx%dx%dx%d"%(int(input_h), self.in_chan, int(after_h), self.out_chan),
                                           'cpu')
        latency_cost = [0.01] + [irb_latency] + [0.01] * (NetworkBlock.state_num - 2)

        if NetworkBlock.device_num > 1:
            irb_latency = \
                NetworkBlock.proximate_latency(irb_name,
                                               "%dx%dx%dx%d" % (int(input_h), self.in_chan, int(after_h), self.out_chan),
                                               'gpu')
            latency_cost_gpu = [0.01] + [irb_latency] + [0.01] * (NetworkBlock.state_num - 2)
            return [latency_cost, latency_cost_gpu]

        return latency_cost


class ASPPBlock(NetworkBlock):
    n_layers = 0
    n_comp_steps = 0
    
    def __init__(self, in_chan, depth, atrous_rates):
        super(ASPPBlock, self).__init__()
        self.atrous_rates = atrous_rates
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        # 1.step
        self.conv_1_step = nn.Conv2d(in_chan, depth, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(depth,
                                  momentum=1.0 if not NetworkBlock.bn_moving_momentum else 0.1,
                                  track_running_stats=NetworkBlock.bn_track_running_stats)
        
        # 2.step
        self.conv_2_step = nn.Conv2d(in_chan, depth, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(depth,
                                  momentum=1.0 if not NetworkBlock.bn_moving_momentum else 0.1,
                                  track_running_stats=NetworkBlock.bn_track_running_stats)
        
        # 3.step
        self.atrous_conv_list = nn.ModuleList([])
        for i, rate in enumerate(self.atrous_rates):
            self.atrous_conv_list.append(SepConvBN(in_chan, depth, relu=True, k_size=3, dilation=rate))
        
        # 5.step
        self.conv_5_step = nn.Conv2d((len(self.atrous_rates) + 2) * depth, depth, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(depth,
                                  momentum=1.0 if not NetworkBlock.bn_moving_momentum else 0.1,
                                  track_running_stats=NetworkBlock.bn_track_running_stats)
        
        self.params = {
            'module_list': ['ASPPBlock'],
            'name_list': ['ASPPBlock'],
            'ASPPBlock': {'in_chan': in_chan,
                          'depth': depth,
                          'atrous_rates': atrous_rates},
            
            'in_chan': in_chan,
            'out_chan': depth
        }
        self.depth = depth
        self.structure_fixed = False
    
    def forward(self, x, sampling=None):
        h = x.shape[2]
        w = x.shape[3]
        branch_logits = []
        
        # 1.step global pooling
        feature_1 = self.global_pool(x)
        feature_1 = self.conv_1_step(feature_1)
        feature_1 = self.bn1(feature_1)
        feature_1 = F.relu(feature_1)
        # feature_1 = F.upsample(feature_1, size=[h, w])
        feature_1 = torch.nn.functional.interpolate(feature_1,size=[h,w],mode='bilinear',align_corners=True)

        branch_logits.append(feature_1)
        
        # 2.step 1x1 convolution
        feature_2 = self.conv_2_step(x)
        feature_2 = self.bn2(feature_2)
        feature_2 = F.relu(feature_2)
        branch_logits.append(feature_2)
        
        # 3.step 3x3 convolutions with different atrous rates
        for i in range(len(self.atrous_conv_list)):
            f = self.atrous_conv_list[i](x)
            branch_logits.append(f)
        
        # 4.step concat
        concat_logits = torch.cat(branch_logits, 1)
        concat_logits = self.conv_5_step(concat_logits)
        concat_logits = self.bn5(concat_logits)
        concat_logits = F.relu(concat_logits)
        
        if sampling is None:
            return concat_logits
        
        is_activate = (int)(sampling.item())
        if is_activate == 1:
            return concat_logits
        else:
            return torch.zeros((x.size(0), self.depth, x.size(2), x.size(3)), device=x.device)
    
    def get_param_num(self, x):
        conv1_params = self.conv_1_step.kernel_size[0] * self.conv_1_step.kernel_size[
            1] * self.conv_1_step.in_channels * self.conv_1_step.out_channels
        conv2_params = self.conv_2_step.kernel_size[0] * self.conv_2_step.kernel_size[
            1] * self.conv_2_step.in_channels * self.conv_2_step.out_channels
        atrous_conv_params = 0
        for index in range(len(self.atrous_conv_list)):
            atrous_conv_params += self.atrous_conv_list[index].get_param_num(x)[1]
        conv5_params = self.conv_5_step.kernel_size[0] * self.conv_5_step.kernel_size[
            1] * self.conv_5_step.in_channels * self.conv_5_step.out_channels
        
        params = conv1_params + conv2_params + atrous_conv_params + conv5_params
        return [0] + [params] + [0] * (NetworkBlock.state_num - 2)
    
    def get_flop_cost(self, x):
        flops = self.get_conv2d_flops(self.conv_1_step, torch.Size((1, x.shape[1], 1, 1)),
                                      torch.Size((1, self.depth, 1, 1)))
        flops += self.get_bn_flops(self.bn1, torch.Size((1, self.depth, 1, 1)), torch.Size((1, self.depth, 1, 1)))
        flops += self.get_relu_flops(None, torch.Size((1, self.depth, 1, 1)), torch.Size((1, self.depth, 1, 1)))
        
        flops += self.get_conv2d_flops(self.conv_2_step,
                                       torch.Size((1, x.shape[1], x.shape[2], x.shape[3])),
                                       torch.Size((1, self.depth, x.shape[2], x.shape[3])))
        flops += self.get_bn_flops(self.bn2, torch.Size((1, self.depth, x.shape[2], x.shape[3])),
                                   torch.Size((1, self.depth, x.shape[2], x.shape[3])))
        flops += self.get_relu_flops(None, torch.Size((1, self.depth, x.shape[2], x.shape[3])),
                                     torch.Size((1, self.depth, x.shape[2], x.shape[3])))
        
        for i in range(len(self.atrous_conv_list)):
            flops += self.atrous_conv_list[i].get_flop_cost(x)[1]
        
        flops += self.get_conv2d_flops(self.conv_5_step,
                                       torch.Size(
                                           (1, (len(self.atrous_rates) + 2) * self.depth, x.shape[2], x.shape[3])),
                                       torch.Size((1, self.depth, x.shape[2], x.shape[3])))
        flops += self.get_bn_flops(self.bn5, torch.Size((1, self.depth, x.shape[2], x.shape[3])),
                                   torch.Size((1, self.depth, x.shape[2], x.shape[3])))
        flops += self.get_relu_flops(None, torch.Size((1, self.depth, x.shape[2], x.shape[3])),
                                     torch.Size((1, self.depth, x.shape[2], x.shape[3])))
        
        return [0] + [flops] + [0] * (NetworkBlock.state_num - 2)


class LargekernelConv(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, in_chan, out_chan, k_size=3, bias=True):
        super(LargekernelConv, self).__init__()

        self.params = {
            'module_list': ['LargekernelConv'],
            'name_list': ['LargekernelConv'],
            'LargekernelConv': {'in_chan': in_chan,
                                'out_chan': out_chan,
                                'k_size': k_size,
                                'bias': bias},
            'in_chan': in_chan,
            'out_chan': out_chan
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

        self.conv1 = nn.Conv2d(out_chan,
                               out_chan,
                               kernel_size=3,
                               padding=1,
                               bias=True)
        self.conv2 = nn.Conv2d(out_chan,
                               out_chan,
                               kernel_size=3,
                               padding=1,
                               bias=True)

        self.in_chan = in_chan
        self.out_chan = out_chan
        self.structure_fixed = False

    def forward(self, x, sampling=None):
        left_x1 = self.left_conv1(x)
        left_x2 = self.left_conv2(left_x1)
        right_x1 = self.right_conv1(x)
        right_x2 = self.right_conv2(right_x1)
        x = left_x2 + right_x2

        x_res = self.conv1(x)
        x_res = F.relu(x_res)
        x_res = self.conv2(x_res)

        x = x + x_res

        if sampling is None:
            return x

        is_activate = (int)(sampling.item())
        if is_activate == 1:
            return x
        else:
            return torch.zeros(x.shape, device=x.device)

    def get_param_num(self, x):
        left_conv1_params = self.left_conv1.kernel_size[0]*self.left_conv1.kernel_size[1]*self.left_conv1.in_channels*self.left_conv1.out_channels
        left_conv2_params = self.left_conv2.kernel_size[0]*self.left_conv2.kernel_size[1]*self.left_conv2.in_channels*self.left_conv2.out_channels
        right_conv1_params = self.right_conv1.kernel_size[0]*self.right_conv1.kernel_size[1]*self.right_conv1.in_channels*self.right_conv1.out_channels
        right_conv2_params = self.right_conv2.kernel_size[0]*self.right_conv2.kernel_size[1]*self.right_conv2.in_channels*self.right_conv2.out_channels

        conv1_params = self.conv1.kernel_size[0]*self.conv1.kernel_size[1]*self.conv1.in_channels*self.conv1.out_channels
        conv2_params = self.conv2.kernel_size[0]*self.conv2.kernel_size[1]*self.conv2.in_channels*self.conv2.out_channels

        params = left_conv1_params+\
                 left_conv2_params+\
                 right_conv1_params+\
                 right_conv2_params+\
                 conv1_params+\
                 conv2_params

        return [0] + [params] + [0]*(NetworkBlock.state_num - 2)

    def get_flop_cost(self, x):
        conv_in_data_size = torch.Size([1, *x.shape[1:]])
        conv_out_data_size = torch.Size([1, self.out_chan, x.shape[-1], x.shape[-1]])

        flops_1 = self.get_conv2d_flops(self.left_conv1, conv_in_data_size, conv_out_data_size)
        flops_2 = self.get_conv2d_flops(self.left_conv2, conv_out_data_size, conv_out_data_size)
        flops_3 = self.get_conv2d_flops(self.right_conv1, conv_in_data_size, conv_out_data_size)
        flops_4 = self.get_conv2d_flops(self.right_conv2, conv_out_data_size, conv_out_data_size)

        flops_5 = 0
        flops_5 += self.get_conv2d_flops(self.conv1, conv_out_data_size, conv_out_data_size)
        flops_5 += conv_out_data_size.numel() / conv_out_data_size[0]
        flops_5 += self.get_conv2d_flops(self.conv2, conv_out_data_size, conv_out_data_size)
        
        flop_cost = flops_1+flops_2+flops_3+flops_4+flops_5
        return [0] + [flop_cost] + [0]*(self.state_num - 2)


class Fused(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, in_chan, out_chan, expand_factor, relu, k_size=3, stride=1, dilation=1):
        super(Fused, self).__init__()
        middle_channels = _make_divisible(in_chan * expand_factor)
        self.expand_conv = nn.Conv2d(in_channels=in_chan,
                                     out_channels=middle_channels,
                                     kernel_size=k_size,
                                     stride=stride,
                                     padding=k_size // 2 + (dilation - 1) * (k_size - 1) // 2,
                                     dilation=dilation,
                                     bias=False)
        self.expand_bn = nn.BatchNorm2d(num_features=middle_channels,
                                        momentum=1.0 if not NetworkBlock.bn_moving_momentum else 0.1,
                                        track_running_stats=NetworkBlock.bn_track_running_stats)

        self.pointwise_conv = nn.Conv2d(in_channels=middle_channels,
                                        out_channels=out_chan,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=False)
        self.pointwise_bn = nn.BatchNorm2d(num_features=out_chan,
                                           momentum=1.0 if not NetworkBlock.bn_moving_momentum else 0.1,
                                           track_running_stats=NetworkBlock.bn_track_running_stats)
        self.relu = relu
        self.out_chan = out_chan

        self.params = {
            'module_list': ['Fused'],
            'name_list': ['Fused'],
            'Fused': {'stride': stride,
                      'out_chan': out_chan,
                      'k_size': k_size,
                      'relu': relu,
                      'dilation': dilation,
                      'in_chan': in_chan},
            'in_chan': in_chan,
            'out_chan': out_chan
        }
        self.structure_fixed = False

    def forward(self, x, sampling=None):
        x = self.expand_conv(x)
        x = self.expand_bn(x)
        x = F.relu(x)

        x = self.pointwise_conv(x)
        x = self.pointwise_bn(x)
        if self.relu:
            x = F.relu(x)

        if sampling is None:
            return x

        is_activate = int(sampling.item())
        if is_activate == 1:
            return x
        else:
            return torch.zeros(x.shape, device=x.device)

    def get_param_num(self, x):
        part1_params = self.expand_conv.kernel_size[0] * self.expand_conv.kernel_size[1] * self.expand_conv.in_channels * self.expand_conv.out_channels
        part2_params = self.pointwise_conv.in_channels * self.pointwise_conv.out_channels
        return [0] + [part1_params + part2_params] + [0] * (NetworkBlock.state_num - 2)

    def get_flop_cost(self, x):
        in_data_size = torch.Size([1, *x.shape[1:]])
        middle_data_size = torch.Size([1,
                                       self.expand_conv.out_channels,
                                       x.shape[-1]//self.expand_conv.stride[0],
                                       x.shape[-1]//self.expand_conv.stride[1]])
        out_data_size = torch.Size([1,
                                    self.out_chan,
                                    x.shape[-1] // self.expand_conv.stride[0],
                                    x.shape[-1] // self.expand_conv.stride[1]])

        flops_1 = self.get_conv2d_flops(self.expand_conv, in_data_size, middle_data_size)
        flops_2 = self.get_bn_flops(self.expand_bn, middle_data_size, middle_data_size)
        flops_3 = self.get_relu_flops(None, middle_data_size, middle_data_size)

        flops_4 = self.get_conv2d_flops(self.pointwise_conv, middle_data_size, out_data_size)
        flops_5 = self.get_bn_flops(self.bn, out_data_size, out_data_size)
        flops_6 = 0
        if self.relu:
            flops_6 = self.get_relu_flops(None, out_data_size, out_data_size)

        total_flops = flops_1 + flops_2 + flops_3 + flops_4 + flops_5 + flops_6
        flop_cost = [0] + [total_flops] + [0] * (self.state_num - 2)
        return flop_cost
