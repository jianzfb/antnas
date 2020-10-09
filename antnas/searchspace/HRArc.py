# -*- coding: UTF-8 -*-
# @Time    : 2020/9/20 1:42 下午
# @File    : HRArc.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antnas.networks.SuperNetwork import *
from antnas.utils.drawers.NASDrawer import NASDrawer
import torch
import torch.nn as nn
from antnas.searchspace.Arc import *

class HRArc(Arc):
    def __init__(self,cell_cls, C, graph):
        super(HRArc, self).__init__(graph)
        self.cell_cls = cell_cls
        self.C = C

    def generate(self, head, tail):
        return None