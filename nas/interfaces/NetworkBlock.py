import timeit

import torch.nn.functional as F
from torch import nn
import torch


class NetworkBlock(nn.Module):
    state_num = 5
    epoch = 0
    is_training = True
    is_running = False

    def __init__(self):
        super(NetworkBlock, self).__init__()
        self._sampling = None

    def get_exec_time(self, x):
        n_exec, time = timeit.Timer(lambda: self(x)).autorange()
        mean_time = time / n_exec
        return mean_time

    @staticmethod
    def get_conv2d_flops(m, x, y):
        assert x.dim() == 4 and y.dim() == 4
        # return x.size(1) * y.size(1) * y.size(2) * y.size(3) * k_size[0] * k_size[1] / (s_size[0] * s_size[1])

        cin = m.in_channels
        cout = m.out_channels
        kh, kw = m.kernel_size
        batch_size = x.size()[0]

        out_h = y.size(2)
        out_w = y.size(3)

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
    def get_bn_flops(m, x,y=None):
        nelements = x.size(0) * x.size(1) * x.size(2) * x.size(3)
        total_ops = 4 * nelements
        return total_ops

    @staticmethod
    def get_relu_flops(m, x,y=None):
        nelements = x.size(0) * x.size(1) * x.size(2) * x.size(3)
        total_ops = nelements
        return total_ops

    @staticmethod
    def get_avgpool_flops(m, x, y):
        total_add = m.kernel_size[0] * m.kernel_size[1]
        total_div = 1

        kernel_ops = total_add + total_div
        num_elements = y.size(0) * y.size(1) * y.size(2) * y.size(3)
        total_ops = kernel_ops * num_elements
        return total_ops

    @staticmethod
    def get_avgglobalpool_flops(m, x, y):
        total_add = (x.size(2) // m.output_size[0]) * (x.size(3) // m.output_size[1])
        total_div = 1

        kernel_ops = total_add + total_div
        num_elements = y.size(0) * y.size(1) * y.size(2) * y.size(3)
        total_ops = kernel_ops * num_elements
        return total_ops

    @staticmethod
    def get_linear_flops(m, x, y):
        total_mul = m.in_features
        total_add = m.in_features - 1
        num_elements = y.numel()
        total_ops = (total_mul + total_add) * num_elements

        return total_ops

    def sampling(self, val):
        self._sampling = val


class DummyBlock(NetworkBlock):
    n_layers = 0
    n_comp_steps = 0

    def __init__(self):
        super(DummyBlock, self).__init__()

    def forward(self, x):
        if self._sampling is None:
            return x
        return x * (self._sampling == 1).float()

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

    def forward(self, x):
        x_res = x
        if self.out_channels > self.in_channels:
            x_res = torch.cat([x, torch.zeros(x.size(0),
                                              (self.out_channels-self.in_channels),
                                              x.size(2),
                                              x.size(3))], dim=1)
        #device=torch.device('cuda')
        if self.reduction:
            x_res = self.pool2d(x_res)

        return x_res

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

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = F.relu(x)

        if self._sampling is None:
            return x
        return x * (self._sampling == 1).float()

    def get_flop_cost(self, x):
        y = self(x)

        flops_1 = self.get_conv2d_flops(self.conv, x, y)
        flops_2 = self.get_bn_flops(self.bn, x, y)
        flops_3 = 0
        if self.relu:
            flops_3 = self.get_relu_flops(self.relu, x, y)

        total_flops = flops_1 + flops_2 + flops_3
        return [0] + [total_flops] + [0] * (self.state_num - 2)


class Upsamp_Block(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, in_chan, out_chan, relu, k_size, bias, scale_factor=2):
        super(Upsamp_Block, self).__init__()
        self.conv_layer = ConvBn(in_chan, out_chan, relu=relu, k_size=k_size, bias=bias)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = F.upsample(x, scale_factor=self.scale_factor)
        x = self.conv_layer(x)

        if self._sampling is None:
            return x

        return x * (self._sampling == 1).float()

    def get_flop_cost(self, x):
        cost = self.conv_layer.get_flop_cost(x)
        return cost


class Add_Block(NetworkBlock):
    n_layers = 0
    n_comp_steps = 1

    def __init__(self):
        super(Add_Block, self).__init__()
        self.history_mean = None
        self.history_var = None

    def forward(self, x):
        if not isinstance(x, list):
            return x
        assert isinstance(x, list)
        return sum(x)

    def get_flop_cost(self, x):
        if not isinstance(x, list):
            return [0] * self.state_num
        assert isinstance(x, list)
        return [0] + [x[0].numel() * (len(x) - 1)] * (self.state_num - 1)


class Moving_Add_Block(NetworkBlock):
    n_layers = 0
    n_comp_steps = 1

    def __init__(self):
        super(Moving_Add_Block, self).__init__()
        self.history_mean = None
        self.history_var = None

    def forward(self, x):
        block_val = x
        if not isinstance(x, list):
            block_val = [x]
        assert isinstance(block_val, list)

        shift_x = sum(block_val)

        if Moving_Add_Block.is_running:
            leaf_shift_x = shift_x.detach()
            x_mean = leaf_shift_x.mean(dim=0, keepdim=True)
            x_mean = x_mean.mean(dim=2, keepdim=True)
            x_mean = x_mean.mean(dim=3, keepdim=True)

            ss = leaf_shift_x.permute(1,0,2,3)
            mm = ss.reshape(ss.size()[0], -1)

            x_var = mm.var(dim=1, keepdim=False)
            x_var[torch.isnan(x_var)] = 0
            x_var = x_var + 1.0
            x_var = x_var.reshape((1, ss.size()[0], 1, 1))

            if Moving_Add_Block.is_training:
                decay = 0.5
                if self.history_mean is None:
                    self.history_mean = x_mean
                    self.history_var = x_var
                else:
                    if Moving_Add_Block.epoch < 50:
                        decay = 0.5
                    elif Moving_Add_Block.epoch < 100:
                        decay = 0.8
                    else:
                        decay = 1.0

                self.history_mean = (1 - decay) * self.history_mean + decay * x_mean
                self.history_mean.detach_()

                self.history_var = (1 - decay) * self.history_var + decay * x_var
                self.history_var.detach_()

            shift_x = torch.div((shift_x - x_mean), x_var) * self.history_var + self.history_mean
        return shift_x

    def get_flop_cost(self, x):
        if not isinstance(x, list):
            return [0] * self.state_num
        assert isinstance(x, list)
        return [0] + [x[0].numel() * (len(x) - 1)] * (self.state_num - 1)


# new layer
# Inverted residual block(mobilenet-v2)
class InvertedResidualBlock(NetworkBlock):
    n_layers = 3
    n_comp_steps = 1

    def __init__(self, in_chan, expansion, kernel_size, out_chan, skip=True, reduction=False):
        super(InvertedResidualBlock, self).__init__()
        # expansion,
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
        y1 = self.conv1(x)
        y2 = self.dwconv2(y1)
        y3 = self.conv3(y2)

        flops_1 = self.get_conv2d_flops(self.conv1, x,y1)
        flops_2 = self.get_bn_flops(self.bn1, x,y1)
        flops_3 = self.get_relu_flops(self.relu1, x, y1)

        flops_4 = self.get_conv2d_flops(self.dwconv2, y1, y2)
        flops_5 = self.get_bn_flops(self.bn2, y1,y2)
        flops_6 = self.get_relu_flops(self.relu2, y1, y2)

        flops_7 = self.get_conv2d_flops(self.conv3, y2, y3)
        flops_8 = self.get_bn_flops(self.bn3, y2, y3)

        flops_9 = 0
        if self.skip and self.in_chan == self.out_chan and (not self.reduction):
            flops_9 = x.size(0) * x.size(1) * x.size(2) * x.size(3)

        total_flops = flops_1 + \
                      flops_2 + \
                      flops_3 + \
                      flops_4 + \
                      flops_5 + \
                      flops_6 + \
                      flops_7 + \
                      flops_8 + flops_9
        return total_flops


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
        x = F.mul(se_x, x)

        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.skip and self.in_chan == self.out_chan and (not self.reduction):
            x = x + input

        return x

    def get_flop_cost(self, x):
        y1 = self.conv1(x)
        y2 = self.dwconv2(y1)
        y3 = self.conv3(y2)

        flops_1 = self.get_conv2d_flops(self.conv1, x, y1)
        flops_2 = self.get_bn_flops(self.bn1, x, y1)
        flops_3 = self.get_relu_flops(self.relu1, x, y1)

        flops_4 = self.get_conv2d_flops(self.dwconv2, y1, y2)
        flops_5 = self.get_bn_flops(self.bn2, y1, y2)
        flops_6 = self.get_relu_flops(self.relu2, y1, y2)

        flops_7 = self.get_conv2d_flops(self.conv3, y2, y3)
        flops_8 = self.get_bn_flops(self.bn3, y2, y3)

        se_y2 = self.global_pool(y2)
        flops_se_1 = self.get_avgglobalpool_flops(self.global_pool, y2, se_y2)

        se_y3 = self.dense_layer_1(se_y2.view(se_y2.size(0), se_y2.size(1)))
        flops_se_2 = self.get_linear_flops(self.dense_layer_1, se_y2 , se_y3)
        flops_se_3 = self.get_relu_flops(F.relu, se_y3.view(se_y3.size(0), se_y3.size(1), 1, 1), se_y3.view(se_y3.size(0), se_y3.size(1), 1, 1))

        se_y4 = self.dense_layer_2(se_y3)
        flops_se_4 = self.get_linear_flops(self.dense_layer_2, se_y3, se_y4)
        flops_se_5 = se_y3.size(0) * se_y3.size(1) * 2
        flops_se_6 = y2.size(0) * y2.size(1) * y2.size(2) * y2.size(3)

        flops_9 = 0
        if self.skip and self.in_chan == self.out_chan and (not self.reduction):
            flops_9 = x.size(0) * x.size(1) * x.size(2) * x.size(3)

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

        return total_flops


class SepConv(NetworkBlock):
    n_layers = 2
    n_comp_steps = 1

    def __init__(self, in_chan, kernel_size, skip=True):
        super(SepConv, self).__init__()

        self.dwconv1 = nn.Conv2d(in_chan,
                                 in_chan,
                                 kernel_size=kernel_size,
                                 stride=1,
                                 groups=in_chan,
                                 padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(in_chan)
        self.relu1 = F.relu6

        self.conv2 = nn.Conv2d(in_chan,
                               in_chan,
                               kernel_size=1,
                               stride=1)
        self.bn2 = nn.BatchNorm2d(in_chan)
        self.relu2 = F.relu6

        self.kernel_size = kernel_size
        self.skip = skip

    def forward(self, input):
        x = input

        x = self.dwconv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        if self.skip:
            x = x + input

        return x

    def get_flop_cost(self, x):
        H = x.size(2)
        W = x.size(3)

        dwconv1_cost = self.dwconv1.in_channels * H * W * self.dwconv1.kernel_size[0] * self.dwconv1.kernel_size[1]
        conv2_cost = self.conv2.in_channels*self.conv2.out_channels*H*W*self.conv2.kernel_size[0]*self.conv2.kernel_size[1]
        return dwconv1_cost+conv2_cost
