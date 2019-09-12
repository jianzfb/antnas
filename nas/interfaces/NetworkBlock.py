import torch.nn.functional as F
from torch import nn
import torch
import json
import os
import threading


class NetworkBlock(nn.Module):
    state_num = 5
    epoch = 0
    lookup_table = {}

    def __init__(self):
        super(NetworkBlock, self).__init__()
        self._sampling = threading.local()
        self._last_sampling = threading.local()
        self.node_regularizer = threading.local()
        self._is_switch = False

    @property
    def switch(self):
        return self._is_switch

    @switch.setter
    def switch(self, val):
        self._is_switch = val

    def get_latency(self, x):
        return [0.0] * NetworkBlock.state_num

    def get_param_num(self, x):
        return [0] * NetworkBlock.state_num

    def get_flop_cost(self, x):
        return [0.0] * self.state_num

    @staticmethod
    def load_lookup_table(lookup_table_file):
        if os.path.exists(lookup_table_file):
            with open(lookup_table_file) as fp:
                NetworkBlock.lookup_table = json.load(fp)
            return True
        else:
            return False

    @staticmethod
    def proximate_latency(kernel_latency, kernel_profile):
        if kernel_profile in kernel_latency:
            return kernel_latency[kernel_profile]

        name_list = []
        latency_list = []
        for k,v in kernel_latency.items():
            a,b,c,d = k.split('x')
            total = (int(a)*int(b)*int(c))/int(d[-1])
            if total not in name_list:
                name_list.append(total)
                latency_list.append(v)

        kernel_a,kernel_b,kernel_c,kernel_d = kernel_profile.split('x')
        kernel_total = (int(kernel_a)*int(kernel_b)*int(kernel_c))/int(kernel_d[-1])

        most_prox_index = 0
        most_prox_val = 100000000000
        for index in range(len(name_list)):
            a = name_list[index] if name_list[index] > kernel_total else kernel_total
            b = kernel_total if name_list[index] >= kernel_total else name_list[index]
            ratio = float(a) / float(b)

            if most_prox_val > ratio:
                most_prox_val = ratio
                most_prox_index = index

        prox_latency = 0.0
        if name_list[most_prox_index] > kernel_total:
            prox_latency = latency_list[most_prox_index] / most_prox_val
        else:
            prox_latency = latency_list[most_prox_index] * most_prox_val
        return prox_latency

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

    def set_sampling(self, val):
        self._sampling.value = val

    def get_sampling(self):
        return getattr(self._sampling, 'value', None)

    def set_last_sampling(self, val):
        self._last_sampling.value = val

    def get_last_sampling(self):
        return getattr(self._last_sampling, 'value', None)

    def set_node_regularizer(self, val):
        self.node_regularizer.value = val

    def get_node_regularizer(self):
        return getattr(self.node_regularizer, 'value', None)


class Identity(NetworkBlock):
    n_layers = 0
    n_comp_steps = 0

    def __init__(self):
        super(Identity, self).__init__()
        self.switch = True

        self.params = {
            'module_list': ['identity'],
            'identity': {}
        }

    def forward(self, x):
        if self.get_sampling() is None:
            return x

        return x * (self._sampling.value == 1).float()

    def get_flop_cost(self, x):
        return [0] * self.state_num


class Skip(NetworkBlock):
    n_layers = 0
    n_comp_steps = 0

    def __init__(self, in_channels, out_channels, reduction):
        super(Skip, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction = reduction
        self.pool2d = torch.nn.AvgPool2d(2, 2)

        self.switch = True

        self.params = {
            'module_list': ['skip'],
            'skip': {'out_chan': out_channels, 'reduction': reduction},
        }

    def forward(self, x):
        x_res = x
        if self.out_channels > self.in_channels:
            x_res = torch.cat([x, torch.zeros(x.size(0),
                                              (self.out_channels-self.in_channels),
                                              x.size(2),
                                              x.size(3), device=x.device)], dim=1)
        if self.reduction:
            x_res = self.pool2d(x_res)

        if self.get_sampling() is None:
            return x_res

        return x_res * (self._sampling.value == 1).float()

    def get_flop_cost(self, x):
        return [0] * self.state_num


class ConvBn(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, in_chan, out_chan, relu, k_size=3, stride=1, padding=1, bias=True):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=k_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = relu
        self.out_chan = out_chan
        self.params = {
            'module_list': ['convbn'],
            'convbn': {'stride': stride,
                       'out_chan': out_chan,
                       'k_size': k_size,
                       'relu': relu,
                       'bias': bias}
        }
        self.switch = True

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = F.relu(x)

        if self.get_sampling() is None:
            return x

        return x * (self._sampling.value == 1).float()

    def get_flop_cost(self, x):
        conv_in_data_size = torch.Size([1, *x.shape[1:]])
        conv_out_data_size = torch.Size([1, self.out_chan, x.shape[-1]//self.conv.stride[0],x.shape[-1]//self.conv.stride[1]])

        flops_1 = self.get_conv2d_flops(self.conv, conv_in_data_size, conv_out_data_size)
        flops_2 = self.get_bn_flops(self.bn, conv_out_data_size, conv_out_data_size)
        flops_3 = 0
        if self.relu:
            flops_3 = self.get_relu_flops(None, conv_out_data_size, conv_out_data_size)

        total_flops = flops_1 + flops_2 + flops_3
        flop_cost = [0] + [total_flops] + [0] * (self.state_num - 2)
        return flop_cost


# class Upsamp_Block(NetworkBlock):
#     n_layers = 1
#     n_comp_steps = 1
#
#     def __init__(self, in_chan, out_chan, relu, k_size, bias, scale_factor=2):
#         super(Upsamp_Block, self).__init__()
#         self.conv_layer = ConvBn(in_chan, out_chan, relu=relu, k_size=k_size, bias=bias)
#         self.scale_factor = scale_factor
#
#     def forward(self, x):
#         x = F.upsample(x, scale_factor=self.scale_factor)
#         x = self.conv_layer(x)
#
#         if self._sampling is None:
#             return x
#
#         return x * (self._sampling == 1).float()
#
#     def get_flop_cost(self, x):
#         cost = self.conv_layer.get_flop_cost(x)
#         return cost


class AddBlock(NetworkBlock):
    n_layers = 0
    n_comp_steps = 1

    def __init__(self):
        super(AddBlock, self).__init__()
        self.params = {
            'module_list': ['addblock'],
            'addblock': {}
        }

    def forward(self, x):
        if not isinstance(x, list):
            return x

        assert isinstance(x, list)
        return sum(x)

    def get_flop_cost(self, x):
        if not isinstance(x, list):
            return [0] * self.state_num

        flop_cost = [0] + [x[0].size().numel()/x[0].size()[0] * (len(x) - 1)] * (self.state_num - 1)
        return flop_cost


# class Moving_Add_Block(NetworkBlock):
#     n_layers = 0
#     n_comp_steps = 1
#
#     def __init__(self):
#         super(Moving_Add_Block, self).__init__()
#         self.history_mean = None
#         self.history_var = None
#
#     def forward(self, x):
#         block_val = x
#         if not isinstance(x, list):
#             block_val = [x]
#         assert isinstance(block_val, list)
#
#         shift_x = sum(block_val)
#
#         if Moving_Add_Block.is_running:
#             leaf_shift_x = shift_x.detach()
#             x_mean = leaf_shift_x.mean(dim=0, keepdim=True)
#             x_mean = x_mean.mean(dim=2, keepdim=True)
#             x_mean = x_mean.mean(dim=3, keepdim=True)
#
#             ss = leaf_shift_x.permute(1,0,2,3)
#             mm = ss.reshape(ss.size()[0], -1)
#
#             x_var = mm.var(dim=1, keepdim=False)
#             x_var[torch.isnan(x_var)] = 0
#             x_var = x_var + 1.0
#             x_var = x_var.reshape((1, ss.size()[0], 1, 1))
#
#             if Moving_Add_Block.is_training:
#                 decay = 0.5
#                 if self.history_mean is None:
#                     self.history_mean = x_mean
#                     self.history_var = x_var
#                 else:
#                     if Moving_Add_Block.epoch < 50:
#                         decay = 0.5
#                     elif Moving_Add_Block.epoch < 100:
#                         decay = 0.8
#                     else:
#                         decay = 1.0
#
#                 self.history_mean = (1 - decay) * self.history_mean + decay * x_mean
#                 self.history_mean.detach_()
#
#                 self.history_var = (1 - decay) * self.history_var + decay * x_var
#                 self.history_var.detach_()
#
#             shift_x = torch.div((shift_x - x_mean), x_var) * self.history_var + self.history_mean
#         return shift_x
#
#     def get_flop_cost(self, x):
#         if not isinstance(x, list):
#             return [0] * self.state_num
#         assert isinstance(x, list)
#         return [0] + [x[0].numel() * (len(x) - 1)] * (self.state_num - 1)


# Inverted residual block(mobilenet-v2)
class InvertedResidualBlock(NetworkBlock):
    n_layers = 3
    n_comp_steps = 1

    def __init__(self, in_chan, expansion, kernel_size, out_chan, skip=True, reduction=False):
        super(InvertedResidualBlock, self).__init__()
        # expansion
        self.conv1 = nn.Conv2d(in_chan,
                               in_chan*expansion,
                               kernel_size=1,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(in_chan*expansion)
        self.relu1 = F.relu

        self.dwconv2 = nn.Conv2d(in_chan*expansion,
                                 in_chan*expansion,
                                 kernel_size=kernel_size,
                                 groups=in_chan*expansion,
                                 stride=2 if reduction else 1,
                                 padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(in_chan*expansion)
        self.relu2 = F.relu

        self.conv3 = nn.Conv2d(in_chan*expansion,
                               out_chan,
                               kernel_size=1,
                               stride=1)
        self.bn3 = nn.BatchNorm2d(out_chan)

        self.skip = skip
        self.reduction=reduction
        self.kernel_size = kernel_size
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.expansion = expansion

    def forward(self, input):
        x = input
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.dwconv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.skip and self.in_chan == self.out_chan and (not self.reduction):
            x = x + input

        return x

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
        flops_3 = self.get_relu_flops(self.relu1, step_2_in_size, step_2_in_size)

        flops_4 = self.get_conv2d_flops(self.dwconv2, step_2_in_size, step_2_out_size)
        flops_5 = self.get_bn_flops(self.bn2, step_2_out_size, step_2_out_size)
        flops_6 = self.get_relu_flops(self.relu2, step_2_out_size, step_2_out_size)

        flops_7 = self.get_conv2d_flops(self.conv3, step_3_in_size, step_3_out_size)
        flops_8 = self.get_bn_flops(self.bn3, step_3_out_size, step_3_out_size)

        flops_9 = 0
        if self.skip and self.in_chan == self.out_chan and (not self.reduction):
            flops_9 = 1 * step_1_in_size[1] * step_1_in_size[2] * step_1_in_size[3]

        total_flops = flops_1 + \
                      flops_2 + \
                      flops_3 + \
                      flops_4 + \
                      flops_5 + \
                      flops_6 + \
                      flops_7 + \
                      flops_8 + \
                      flops_9

        self.flop_cost = [0] + [total_flops] * (self.state_num - 1)
        return self.flop_cost

    def get_latency(self, x):
        op_latency_table = NetworkBlock.lookup_table['op']

        x_size = x.shape[-1]
        op_1_name = "convbn_1x1"
        op_1_kernel_profile = "%dx%dx%dxS%d"%(x_size,self.conv1.in_channels,self.conv1.out_channels,self.conv1.stride[0])
        op_1_latency = NetworkBlock.proximate_latency(op_latency_table[op_1_name]['latency'], op_1_kernel_profile)

        if self.reduction:
            x_size = x_size / 2

        op_2_name = "depthwise_%dx%d"%(self.dwconv2.kernel_size[0],self.dwconv2.kernel_size[1])
        op_2_kernel_profile = "%dx%dx%dxS%d"%(x_size,self.dwconv2.in_channels,self.dwconv2.out_channels,self.dwconv2.stride[0])
        op_2_latency = NetworkBlock.proximate_latency(op_latency_table[op_2_name]['latency'], op_2_kernel_profile)

        op_3_name = "convbn_1x1"
        op_3_kernel_profile = "%dx%dx%dxS%d"%(x_size,self.conv3.in_channels,self.conv3.out_channels,self.conv3.stride[0])
        op_3_latency = NetworkBlock.proximate_latency(op_latency_table[op_3_name]['latency'], op_3_kernel_profile)

        total_latency = op_1_latency + op_2_latency + op_3_latency
        self.latency_cost = [0] + [total_latency] * (self.state_num - 1)
        return self.latency_cost


class InvertedResidualBlockWithSE(NetworkBlock):
    n_layers = 3
    n_comp_steps = 1

    def __init__(self, in_chan, expansion, kernel_size, out_chan, skip=True, reduction=False, ratio=4):
        super(InvertedResidualBlockWithSE, self).__init__()
        # expansion,
        self.conv1 = nn.Conv2d(in_chan,
                               in_chan * expansion,
                               kernel_size=1,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(in_chan * expansion)
        self.relu1 = F.relu

        self.dwconv2 = nn.Conv2d(in_chan * expansion,
                                 in_chan * expansion,
                                 kernel_size=kernel_size,
                                 groups=in_chan * expansion,
                                 stride=2 if reduction else 1,
                                 padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(in_chan * expansion)

        self.relu2 = F.relu
        self.conv3 = nn.Conv2d(in_chan * expansion,
                               out_chan,
                               kernel_size=1,
                               stride=1)
        self.bn3 = nn.BatchNorm2d(out_chan)

        # for se
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.dense_layer_1 = torch.nn.Linear(in_chan * expansion, (in_chan * expansion)//ratio)
        self.dense_layer_2 = torch.nn.Linear((in_chan * expansion)//ratio, in_chan * expansion)
        self.ratio = ratio

        self.skip = skip
        self.reduction = reduction
        self.kernel_size = kernel_size
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.expansion = expansion

    def forward(self, input):
        x = input
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.dwconv2(x)
        x = self.bn2(x)

        se_x = self.global_pool(x).view(x.size(0), self.in_chan*self.expansion)
        se_x = self.dense_layer_1(se_x)
        se_x = F.relu(se_x)
        se_x = self.dense_layer_2(se_x)

        se_x = (0.2 * se_x) + 0.5
        se_x = F.threshold(-se_x, -1, -1)
        se_x = F.threshold(-se_x, 0, 0)

        se_x = se_x.view(se_x.size(0), self.in_chan*self.expansion, 1, 1)
        x = torch.mul(se_x, x)

        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.skip and self.in_chan == self.out_chan and (not self.reduction):
            x = x + input

        return x

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
        flops_3 = self.get_relu_flops(self.relu1, step_2_in_size, step_2_in_size)

        flops_4 = self.get_conv2d_flops(self.dwconv2, step_2_in_size, step_2_out_size)
        flops_5 = self.get_bn_flops(self.bn2, step_2_out_size, step_2_out_size)
        flops_6 = self.get_relu_flops(self.relu2, step_2_out_size, step_2_out_size)

        flops_7 = self.get_conv2d_flops(self.conv3, step_3_in_size, step_3_out_size)
        flops_8 = self.get_bn_flops(self.bn3, step_3_out_size, step_3_out_size)

        step_2_1_size = torch.Size([1, self.in_chan*self.expansion])
        step_2_2_size = torch.Size([1, (self.in_chan * self.expansion)//self.ratio])
        step_2_3_size = torch.Size([1, self.in_chan * self.expansion])
        flops_se_1 = self.get_avgglobalpool_flops(self.global_pool, step_2_out_size, step_2_1_size)
        flops_se_2 = self.get_linear_flops(self.dense_layer_1, step_2_1_size, step_2_2_size)
        flops_se_3 = self.get_relu_flops(F.relu, step_2_2_size, step_2_2_size)
        flops_se_4 = self.get_linear_flops(self.dense_layer_2, step_2_2_size, step_2_3_size)
        flops_se_5 = 1 * step_2_3_size[1] * 2
        flops_se_6 = 1 * step_2_out_size[1] * step_2_out_size[2] * step_2_out_size[3]

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
                      flops_9 + \
                      flops_se_1 + \
                      flops_se_2 + \
                      flops_se_3 + \
                      flops_se_4 + \
                      flops_se_5 + \
                      flops_se_6

        flop_cost = [0] + [total_flops] * (self.state_num - 1)
        return flop_cost

    def get_latency(self, x):
        return [0] * self.state_num
