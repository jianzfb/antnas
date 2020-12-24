# -*- coding: UTF-8 -*-
# @Time    : 2020/10/21 3:45 下午
# @File    : mobilenetv2.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function


import torch.nn as nn
import math
import ununiformpool
import torch
from torchinterp1d import Interp1d
import numpy as np

__all__ = ['mobilenetv2']


class UniformPoolFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, mask):
        data_tensor = torch.tensor(data)
        mask_tensor = torch.tensor(mask)
        outputlist = ununiformpool.forward(data_tensor, mask_tensor)
        output = outputlist[0]
        pooling_x_region = outputlist[1]
        pooling_y_region = outputlist[2]
        ctx.save_for_backward(pooling_x_region, pooling_y_region)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output = ununiformpool.backward(grad_output, *ctx.saved_variables)[0]
        return output, None


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, using_ununiformpool=False):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.using_ununiformpool = using_ununiformpool
        self.stride = stride

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        self.conv = None

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            if not self.using_ununiformpool:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.ModuleList([])
                # pw
                self.conv.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
                self.conv.append(nn.BatchNorm2d(hidden_dim))
                self.conv.append(nn.ReLU6(inplace=True))
                # dw
                self.conv.append(nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False))
                self.conv.append(nn.BatchNorm2d(hidden_dim))
                self.conv.append(nn.ReLU6(inplace=True))
                # pw-linear
                self.conv.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
                self.conv.append(nn.BatchNorm2d(oup))

                self.se_global_pool = nn.AdaptiveAvgPool2d((6, 6))
                self.se_conv_layer = nn.Conv2d(hidden_dim,
                                                 1,
                                                 kernel_size=1,
                                                 bias=True,
                                                 stride=1,
                                                 padding=0)

    def forward(self, x):
        if self.using_ununiformpool:
            y = self.conv[0](x)
            y = self.conv[1](y)
            y = self.conv[2](y)

            y_se = self.se_global_pool(y)
            y_se = self.se_conv_layer(y_se)
            y_se = torch.sigmoid(y_se)
            y_se = torch.nn.functional.upsample_bilinear(y_se, (y.shape[2], y.shape[3]))
            y = torch.mul(y, y_se)

            #################
            B = y.shape[0]
            C = y.shape[1]
            H = y.shape[2]
            W = y.shape[3]
            target_c = W // 2
            target_r = H // 2
            downsample_y = \
                torch.zeros((B, C, target_r, target_c), dtype=torch.float32, device=y.device)
            for b in range(B):
                # 获得第b个样本
                yb = y[b]               # C,H,W
                yb_se = y_se[b, 0]      # H,W

                # 获得X方向重采样
                row_mask = yb_se + 0.001
                row_mask_sum = torch.sum(row_mask, axis=1)
                row_mask_sum = torch.unsqueeze(row_mask_sum, dim=-1)
                w = (W - W * 0.5) / row_mask_sum
                xx = w * row_mask + 0.5
                xx = torch.cumsum(xx, dim=1)
                xx = xx.repeat(C, 1, 1)
                xx = xx.detach()

                middle_feature = torch.zeros((C, H, target_c), dtype=torch.float32, device=y.device)
                middle_mask = torch.zeros((H, target_c), dtype=torch.float32, device=y.device)
                for r in range(H):
                    check_p = np.tile(np.expand_dims(np.array(list(range(target_c))) * 2, 0), [C, 1])
                    check_p = torch.tensor(check_p, device=y.device)
                    middle_feature[:, r, :] =\
                        Interp1d()(xx[:, r, :], yb[:, r, :], check_p)
                    middle_mask[r] = \
                        Interp1d()(xx[0, r], yb_se[r], torch.tensor(np.array(list(range(target_c))) * 2, device=y.device))

                # 获得Y方向重采样
                col_mask = middle_mask + 0.001
                col_mask_sum = torch.sum(col_mask, axis=0)
                col_mask_sum = torch.unsqueeze(col_mask_sum, 0)
                w = (H - H * 0.5) / col_mask_sum
                yy = w * col_mask + 0.5
                yy = torch.cumsum(yy, axis=0)
                yy = yy.repeat(C, 1, 1)
                yy = yy.detach()

                final_feature = torch.zeros((C, target_r, target_c), dtype=torch.float32, device=y.device)
                for c in range(target_c):
                    check_p = np.tile(np.expand_dims(np.array(list(range(target_r))) * 2, 0), [C, 1])
                    check_p = torch.tensor(check_p, device=y.device)
                    final_feature[:, :, c] = Interp1d()(yy[:, :, c], middle_feature[:, :, c], check_p)

                downsample_y[b] = final_feature
            #################
            y = downsample_y

            y = self.conv[3](y)
            y = self.conv[4](y)
            y = self.conv[5](y)

            y = self.conv[6](y)
            y = self.conv[7](y)
            if self.identity:
                return x + y
            else:
                return y
        else:
            if self.identity:
                return x + self.conv(x)
            else:
                return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        count = 0
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                if i == 0 and s == 2 and count < 2:
                    layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, True))
                    input_channel = output_channel
                else:
                    layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                    input_channel = output_channel
            count += 1

        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenetv2(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2(**kwargs)