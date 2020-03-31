# -*- coding: UTF-8 -*-
# @Time    : 2019-08-19 18:16
# @File    : NetworkBlock.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import torch.nn.functional as F
from torch import nn
import torch
import json
import os


class NetworkBlock(nn.Module):
    state_num = 5
    bn_moving_momentum = False
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
        return [0.0] * NetworkBlock.state_num

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
    def proximate_latency(op_lookuptable, profile):
        if profile in op_lookuptable['latency']:
            return op_lookuptable['latency'][profile]

        return 0.0

    @staticmethod
    def get_conv2d_flops(m, x_size, y_size):
        # assert x.dim() == 4 and y.dim() == 4
        # return x.size(1) * y.size(1) * y.size(2) * y.size(3) * k_size[0] * k_size[1] / (s_size[0] * s_size[1])

        cin = m.in_channels
        cout = m.out_channels
        kh, kw = m.kernel_size
        batch_size = 1

        out_h = y_size[2]
        out_w = y_size[3]

        # ops per output element
        # kernel_mul = kh * kw * cin
        # kernel_add = kh * kw * cin - 1
        kernel_ops = 1 * kh * kw
        bias_ops = 1 if m.bias is not None else 0
        ops_per_element = kernel_ops + bias_ops

        # total ops
        # num_out_elements = y.numel()
        output_elements = batch_size * out_w * out_h * cout
        total_ops = output_elements * ops_per_element * cin // m.groups
        total_ops += out_w*out_h*cout*(cin*kh*kw-1) // m.groups
        return total_ops

    @staticmethod
    def get_bn_flops(m, x_size, y_size=None):
        nelements = 1 * x_size[1] * x_size[2] * x_size[3]
        total_ops = 4 * nelements
        return total_ops

    @staticmethod
    def get_relu_flops(m, x_size, y_size=None):
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

    def __init__(self):
        super(Identity, self).__init__()
        self.structure_fixed = False

        self.params = {
            'module_list': ['Identity'],
            'name_list': ['Identity'],
            'Identity': {}
        }

    def forward(self, x, sampling=None):
        if sampling is None:
            return x

        return x * (sampling == 1).float()

    def get_flop_cost(self, x):
        return [0] * self.state_num


class Skip(NetworkBlock):
    n_layers = 0
    n_comp_steps = 0

    def __init__(self, in_chan, out_chan, reduction):
        super(Skip, self).__init__()
        self.in_channels = in_chan
        self.out_channels = out_chan
        self.reduction = reduction
        self.pool2d = torch.nn.AvgPool2d(2, 2)
        self.structure_fixed = False

        self.params = {
            'module_list': ['Skip'],
            'name_list': ['Skip'],
            'Skip': {'out_chan': out_chan, 'reduction': reduction},
        }

    def forward(self, x, sampling=None):
        x_res = x
        if self.out_channels > self.in_channels:
            x_res = torch.cat([x, torch.zeros(x.size(0),
                                              (self.out_channels-self.in_channels),
                                              x.size(2),
                                              x.size(3), device=x.device)], dim=1)
        if self.reduction:
            x_res = self.pool2d(x_res)

        if sampling is None:
            return x_res

        return x_res * (sampling == 1).float()

    def get_flop_cost(self, x):
        return [0] * NetworkBlock.state_num

    def get_latency(self, x):
        return [0] * NetworkBlock.state_num


class ConvBn(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, in_chan, out_chan, relu, k_size=3, stride=1, dilation=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=k_size, stride=stride, padding=k_size//2, bias=False, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_chan, momentum=1.0 if not NetworkBlock.bn_moving_momentum else 0.1)
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
                       'in_chan': in_chan}
        }
        self.structure_fixed = False

    def get_param_num(self, x):
        return [0] + [self.conv.kernel_size[0]*self.conv.kernel_size[1]*self.conv.in_channels*self.conv.out_channels] + [0]*(NetworkBlock.state_num - 2)

    def forward(self, x, sampling=None):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = F.relu(x)

        if sampling is None:
            return x

        return x * (sampling == 1).float()

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
        op_latency_table = NetworkBlock.lookup_table['op']
        op_name = "convbn_%dx%d" % (self.conv.kernel_size[0], self.conv.kernel_size[1])

        input_h, _ = x.shape[2:]
        after_h = input_h // self.conv.stride[0]
        op_latency = NetworkBlock.proximate_latency(op_latency_table[op_name],
                                                    '%dx%dx%dx%d'%(int(input_h),
                                                                   self.conv.in_channels,
                                                                   int(after_h),
                                                                   self.conv.out_channels))
        latency_cost = [0] + [op_latency] + [0] * (NetworkBlock.state_num - 2)
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
        self.pointwise_conv = nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_chan, momentum=1.0 if not NetworkBlock.bn_moving_momentum else 0.1)
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
                          'in_chan': in_chan}
        }
        self.structure_fixed = False

    def forward(self, x, sampling=None):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        if self.relu:
            x = F.relu(x)

        if sampling is None:
            return x

        return x * (sampling == 1).float()

    def get_param_num(self, x):
        part1_params = self.depthwise_conv.kernel_size[0]*self.depthwise_conv.kernel_size[1]*self.depthwise_conv.in_channels
        part2_params = self.pointwise_conv.in_channels * self.pointwise_conv.out_channels
        return [0] + [part1_params+part2_params] + [0]*(NetworkBlock.state_num - 2)

    def get_flop_cost(self, x):
        conv_in_data_size = torch.Size([1, *x.shape[1:]])
        conv_out_data_size = torch.Size([1,
                                         self.out_chan,
                                         x.shape[-1]//self.depthwise_conv.stride[0],
                                         x.shape[-1]//self.depthwise_conv.stride[1]])

        flops_1 = self.get_conv2d_flops(self.depthwise_conv, conv_in_data_size, conv_out_data_size)
        flops_2 = self.get_conv2d_flops(self.pointwise_conv, conv_out_data_size, conv_out_data_size)
        flops_3 = self.get_bn_flops(self.bn, conv_out_data_size, conv_out_data_size)
        flops_4 = 0
        if self.relu:
            flops_4 = self.get_relu_flops(None, conv_out_data_size, conv_out_data_size)

        total_flops = flops_1 + flops_2 + flops_3 + flops_4
        flop_cost = [0] + [total_flops] + [0] * (self.state_num - 2)
        return flop_cost

    def get_latency(self, x):
        op_latency_table = NetworkBlock.lookup_table['op']
        op_name = "sepconvbn_%dx%d" % (self.depthwise_conv.kernel_size[0], self.depthwise_conv.kernel_size[1])

        input_h, _ = x.shape[2:]
        after_h = input_h // self.conv.stride[0]
        op_latency = NetworkBlock.proximate_latency(op_latency_table[op_name],
                                                    '%dx%dx%dx%d' % (int(input_h),
                                                                     self.conv.in_channels,
                                                                     int(after_h),
                                                                     self.conv.out_channels))
        latency_cost = [0] + [op_latency] + [0] * (NetworkBlock.state_num - 2)
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
                             'in_chan': in_chan}
        }

    def forward(self, x, sampling=None):
        x = F.upsample(x, scale_factor=self.scale_factor, mode='bilinear')
        if self.conv_layer is not None:
            x = self.conv_layer(x)

        if sampling is None:
            return x

        return x * (sampling == 1).float()

    def get_param_num(self, x):
        if self.conv_layer is not None:
            return [0] + [self.conv_layer.get_param_num(x)[1]] + [0] * (NetworkBlock.state_num - 2)
        else:
            return [0] * NetworkBlock.state_num

    def get_flop_cost(self, x):
        flops = 9 * (x.shape[2]*self.scale_factor)*(x.shape[3]*self.scale_factor) * x.shape[1]
        if self.conv_layer is not None:
            x = F.upsample(x, scale_factor=self.scale_factor, mode='bilinear')
            flops += self.conv_layer.get_flop_cost(x)[1]
        return [0] + [flops] + [0] * (NetworkBlock.state_num - 2)


class AddBlock(NetworkBlock):
    n_layers = 0
    n_comp_steps = 1

    def __init__(self):
        super(AddBlock, self).__init__()
        self.params = {
            'module_list': ['AddBlock'],
            'name_list': ['AddBlock'],
            'AddBlock': {}
        }

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
            'ConcatBlock': {}
        }

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


class InvertedResidualBlockWithSEHS(NetworkBlock):
    n_layers = 3
    n_comp_steps = 1

    def __init__(self,
                 in_chan,
                 expansion,
                 kernel_size,
                 out_chan,
                 skip=True,
                 reduction=False,
                 ratio=4,
                 se=True,
                 hs=True,
                 dilation=1):
        super(InvertedResidualBlockWithSEHS, self).__init__()
        # expansion,
        self.conv1 = nn.Conv2d(in_chan,
                               in_chan * expansion,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(in_chan * expansion, momentum=1.0 if not NetworkBlock.bn_moving_momentum else 0.1)

        self.dwconv2 = nn.Conv2d(in_chan * expansion,
                                 in_chan * expansion,
                                 kernel_size=kernel_size,
                                 groups=in_chan * expansion,
                                 stride=2 if reduction else 1,
                                 padding=kernel_size // 2 + (kernel_size-1)*(dilation-1) // 2,
                                 bias=False,
                                 dilation=dilation)
        self.bn2 = nn.BatchNorm2d(in_chan * expansion,momentum=1.0 if not NetworkBlock.bn_moving_momentum else 0.1)

        # for se
        if se:
            self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.se_conv_layer_1 = nn.Conv2d(in_chan * expansion,
                                             (in_chan * expansion) // ratio,
                                             kernel_size=1,
                                             bias=True,
                                             stride=1,
                                             padding=0)
            self.se_conv_layer_2 = nn.Conv2d((in_chan * expansion) // ratio,
                                             in_chan * expansion,
                                             kernel_size=1,
                                             bias=True,
                                             stride=1,
                                             padding=0)

        self.conv3 = nn.Conv2d(in_chan * expansion,
                               out_chan,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_chan,momentum=1.0 if not NetworkBlock.bn_moving_momentum else 0.1)
        self.ratio = ratio

        self.reduction = reduction
        self.kernel_size = kernel_size
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.expansion = expansion
        self.se = se
        self.hs = hs
        if skip and self.in_chan == self.out_chan and (not self.reduction):
            self.skip = True
        else:
            self.skip = False

    def forward(self, input, sampling=None):
        x = input
        x = self.conv1(x)
        x = self.bn1(x)
        if self.hs:
            x = x * (F.relu6(x + 3.0) / 6.0)
        else:
            x = F.relu(x)

        x = self.dwconv2(x)
        x = self.bn2(x)
        if self.hs:
            x = x * (F.relu6(x + 3.0) / 6.0)
        else:
            x = F.relu(x)

        if self.se:
            se_x = self.global_pool(x)
            se_x = self.se_conv_layer_1(se_x)
            se_x = F.relu(se_x)

            se_x = self.se_conv_layer_2(se_x)
            se_x = F.relu6(se_x + 3.0) / 6.0
            x = torch.mul(se_x, x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.skip and self.in_chan == self.out_chan and (not self.reduction):
            x = x + input

        return x

    def get_param_num(self, x):
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
        step_2_in_size = [1, self.in_chan*self.expansion, x.shape[2], x.shape[3]]
        step_2_out_size = [1, self.in_chan*self.expansion, x.shape[2], x.shape[3]]
        if self.reduction:
            step_2_out_size[2] = step_2_out_size[2] // 2
            step_2_out_size[3] = step_2_out_size[3] // 2
        step_2_in_size = torch.Size(step_2_in_size)
        step_2_out_size = torch.Size(step_2_out_size)

        step_3_in_size = torch.Size([1, step_2_out_size[1], step_2_out_size[2],step_2_out_size[3]])
        step_3_out_size = torch.Size([1, self.out_chan, step_2_out_size[2], step_2_out_size[3]])

        flops_1 = self.get_conv2d_flops(self.conv1, step_1_in_size, step_2_in_size)
        flops_2 = self.get_bn_flops(self.bn1, step_2_in_size, step_2_in_size)
        flops_3 = 0.0
        if self.hs:
            flops_3 = self.get_hs_flops(None, step_2_in_size, step_2_in_size)
        else:
            flops_3 = self.get_relu_flops(F.relu6, step_2_in_size, step_2_in_size)

        flops_4 = self.get_conv2d_flops(self.dwconv2, step_2_in_size, step_2_out_size)
        flops_5 = self.get_bn_flops(self.bn2, step_2_out_size, step_2_out_size)

        flops_6 = 0.0
        if self.hs:
            flops_6 = self.get_hs_flops(None, step_2_out_size, step_2_out_size)
        else:
            flops_6 = self.get_relu_flops(F.relu6, step_2_out_size, step_2_out_size)

        flops_7 = self.get_conv2d_flops(self.conv3, step_3_in_size, step_3_out_size)
        flops_8 = self.get_bn_flops(self.bn3, step_3_out_size, step_3_out_size)

        flops_se = 0.0
        if self.se:
            step_2_1_size = torch.Size([1, self.in_chan*self.expansion, 1, 1])
            step_2_2_size = torch.Size([1, (self.in_chan * self.expansion)//self.ratio, 1, 1])
            step_2_3_size = torch.Size([1, self.in_chan * self.expansion, 1, 1])
            flops_se_1 = self.get_avgglobalpool_flops(self.global_pool, step_2_out_size, step_2_1_size)
            flops_se_2 = self.get_conv2d_flops(self.se_conv_layer_1, step_2_out_size, step_2_1_size)
            flops_se_3 = self.get_relu_flops(F.relu6, step_2_2_size, step_2_2_size)
            flops_se_4 = self.get_conv2d_flops(self.se_conv_layer_2, step_2_2_size, step_2_3_size)

            flops_se_5 = 1 * step_2_3_size[1] * 2
            flops_se_6 = 1 * step_2_out_size[1] * step_2_out_size[2] * step_2_out_size[3]
            flops_se = flops_se_1 + flops_se_2 + flops_se_3 + flops_se_4 + flops_se_5 + flops_se_6

        flops_9 = 0
        if self.skip and self.in_chan == self.out_chan and (not self.reduction):
            flops_9 = 1 * step_3_out_size[1] * step_3_out_size[2] * step_3_out_size[3]

        total_flops = flops_1 + \
                      flops_2 + \
                      flops_3 + \
                      flops_4 + \
                      flops_5 + \
                      flops_6 + \
                      flops_7 + \
                      flops_8 + \
                      flops_9 + flops_se

        flop_cost = [0] + [total_flops] * (self.state_num - 1)
        return flop_cost

    def get_latency(self, x):
        op_latency_table = NetworkBlock.lookup_table['op']

        irb_name = 'irb_%dx%d'%(self.kernel_size, self.kernel_size)
        if self.se:
            irb_name += "_se"
        if self.hs:
            irb_name += "_hs"

        irb_name += "_e%d"%self.expansion
        if self.skip:
            irb_name += "_skip"

        input_h, _ = x.shape[2:]
        after_h = input_h if not self.reduction else input_h // 2
        irb_latency = NetworkBlock.proximate_latency(op_latency_table[irb_name],
                                                     "%dx%dx%dx%d"%(int(input_h), self.in_chan, int(after_h), self.out_chan))
        latency_cost = [0] + [irb_latency] + [0] * (NetworkBlock.state_num - 2)
        return latency_cost


if __name__ == '__main__':
    # 1.step test convbn flops
    # a = ConvBn(128, 256, True, 3)
    # t = torch.ones((1,128,10,10))
    # b = a.get_flop_cost(t)
    # print(b)

    # # 2.step test sepconvbn
    # a = SepConvBN(128,256,True,3)
    # t = torch.ones((1,128,10,10))
    # b = a.get_flop_cost(t)
    # print(b)

    # 3.step test
    pass
