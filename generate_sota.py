# -*- coding: UTF-8 -*-
# @Time    : 2021/1/4 10:31 上午
# @File    : generate_sota.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antnas.searchspace.PKAutoArc import *
from antnas.searchspace.PKArc import *
from antnas.component.NetworkBlock import *
from antnas.component.NetworkBlock import _make_divisible
from sota.mobilenet_v3 import *
import logging


class ImageNetOutLayer(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, in_chan, **kwargs):
        super(ImageNetOutLayer, self).__init__()
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.conv_1 = nn.Conv2d(in_chan, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_1 = nn.BatchNorm2d(960,
                                   momentum=1.0 if not NetworkBlock.bn_moving_momentum else 0.1,
                                   track_running_stats=NetworkBlock.bn_track_running_stats)

        self.conv_2 = nn.Conv2d(960, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.classifier = nn.Linear(1280, 1000)
        self.dropout = torch.nn.Dropout(p=0.9)

        self.params = {
            'module_list': ['ImageNetOutLayer'],
            'name_list': ['ImageNetOutLayer'],
            'ImageNetOutLayer': {'in_chan': in_chan},
            'out': 'outname',
            'in_chan': in_chan,
        }

    def forward(self, x, sampling=None):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = F.relu(x)

        x = self.global_pool(x)
        x = self.conv_2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# mobilenetv3-large
# head (固定计算节点，对应激活参数不可学习)
head = ConvBn(3, 16, k_size=3, stride=1, relu=True)
# tail (固定计算节点，结构不可学习)
tail = ImageNetOutLayer((int)(160*0.75))
mobilenetv3_large(head, tail, "", 0.75)