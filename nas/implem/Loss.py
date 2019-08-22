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
    # old_loss = -F.log_softmax(predictions, dim=1).gather(1, labels.view(-1, 1)).squeeze()
    return F.cross_entropy(predictions, labels, reduce=False)


def segmentation_cross_entropy(predictions, labels):
    bs = predictions.size(0)
    softmax_pred = nn.Softmax2d()(predictions)
    flatten_preds = softmax_pred.view(bs, predictions.size(1), -1)
    flatten_labels = labels.view(bs, 1, -1)
    individual_losses = -torch.log(flatten_preds).gather(1, flatten_labels).view(bs, -1).mean(1)
    return individual_losses
