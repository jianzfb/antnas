# -*- coding: UTF-8 -*-
# @Time    : 2020/10/23 8:55 上午
# @File    : MNv2FixedNetwork.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from antnas.component.NetworkBlock import *
from antnas.component.NetworkCell import *
from antnas.component.Loss import *
import networkx as nx
import copy
from antnas.utils.drawers.NASDrawer import *
from antnas.networks.mobilenetv2 import *


class MNV2FixedNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MNV2FixedNetwork, self).__init__()
        self._accuracy_evaluator_cls = kwargs.get('accuracy_evaluator_cls', None)
        self.blocks = mobilenetv2()

    def forward(self, x, y):
        model_out = self.blocks(x)
        return model_out

    def __str__(self):
        return ''

    def accuracy_evaluator(self):
        return self._accuracy_evaluator_cls()
