# -*- coding: UTF-8 -*-
# @Time    : 2020-05-07 15:03
# @File    : OutLayerFactory.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from nas.component.NetworkBlock import *
from nas.component.NetworkBlock import _make_divisible


class ImageNetOutLayer(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1
    
    def __init__(self, in_chan, out_chan, num_classes):
        super(ImageNetOutLayer, self).__init__()
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_chan,
                                 momentum=1.0 if not NetworkBlock.bn_moving_momentum else 0.1,
                                 track_running_stats=NetworkBlock.bn_track_running_stats)
        
        self.classifier = nn.Linear(out_chan, num_classes)
        
        self.params = {
            'module_list': ['ImageNetOutLayer'],
            'name_list': ['ImageNetOutLayer'],
            'ImageNetOutLayer': {'in_chan': in_chan, 'out_chan': out_chan, 'num_classes': num_classes},
            'out': 'outname',
            'in_chan': in_chan,
            'out_chan': out_chan
        }
    
    def forward(self, x, sampling=None):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Cifar10OutLayer(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, out_shape, in_chan=160):
        super(Cifar10OutLayer, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_shape[0], kernel_size=1, stride=1, padding=0, bias=bias)
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.out_shape = out_shape
        self.params = {
            'module_list': ['Cifar10OutLayer'],
            'name_list': ['Cifar10OutLayer'],
            'Cifar10OutLayer': {'out_shape': out_shape, 'in_chan': in_chan},
            'out': 'outname'
        }

    def forward(self, x, sampling=None):
        x = self.conv(x)
        x = self.global_pool(x)
        return x.view(-1, *self.out_shape)

    def get_flop_cost(self, x):
        return [0] + [0] * (self.state_num - 1)
