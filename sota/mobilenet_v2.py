# -*- coding: UTF-8 -*-
# @Time    : 2020-04-01 12:54
# @File    : mobilenet_v2.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antnas.searchspace.PKAutoArc import *
from antnas.searchspace.PKArc import *
from antnas.component.NetworkBlock import *
from antnas.component.NetworkBlock import _make_divisible
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# mobilenetv2
def mobilenetv2(head, tail, prefix='', width_mult=1.0):
    # setting of inverted residual blocks
    cfgs = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]
    input_channel = head.params['out_chan']
    modules = []
    for t, c, n, s in cfgs:
        output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
        for i in range(n):
            modules.append(InvertedResidualBlockWithSEHS(in_chan=input_channel,
                                                         out_chan=output_channel,
                                                         expansion=t,
                                                         kernel_size=3,
                                                         reduction=True if (i == 0) and (s == 2) else False,
                                                         se=False,
                                                         hs=False))
            input_channel = output_channel

    graph = nx.DiGraph()
    pk = PKAutoArc(graph, None)
    pk.generate(head, tail, modules)
    pk.save('./', '%s_mobilenetv2'%prefix if prefix != '' else 'mobilenetv2')


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


# # cifar mobilenetv3-large
# # head (固定计算节点，对应激活参数不可学习)
# head = ConvBn(3, 16, k_size=3, stride=1, relu=True)
# # tail (固定计算节点，结构不可学习)
# tail = OutLayer(160, (10,), True)
# mobilenetv3_large(head,tail, "./")

# # cifar mobilenetv2
# # head (固定计算节点，对应激活参数不可学习)
# head = ConvBn(3, 32, k_size=3, stride=1, relu=True)
# # tail (固定计算节点，结构不可学习)
# tail = OutLayer(320, (10,), True)
# mobilenetv2(head, tail, 'cifar')


# ImageNet mobilenetv2
width_mult = 1.0
input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
head = ConvBn(3, input_channel, True, 3, 2)
output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
tail = ImageNetOutLayer(320)
mobilenetv2(head, tail)
