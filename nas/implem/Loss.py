# -*- coding: UTF-8 -*-
# @Time    : 2019-08-19 19:28
# @File    : Loss.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import torch.nn.functional as F
from torch import nn
import torch


def cross_entropy(predictions, labels):
    individual_losses = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)(predictions, labels)
    return individual_losses
