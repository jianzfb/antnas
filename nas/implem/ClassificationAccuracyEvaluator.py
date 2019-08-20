# -*- coding: UTF-8 -*-
# @Time    : 2019-08-19 19:27
# @File    : ClassificationAccuracyEvaluator.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from nas.interfaces.AccuracyEvaluator import *
import torch


class ClassificationAccuracyEvaluator(AccuracyEvaluator):
    def __init__(self, *args, **kwargs):
        super(ClassificationAccuracyEvaluator, self).__init__(*args, **kwargs)

    def accuracy(self, preditions, labels):
        _, predicted = torch.max(preditions.data, 1)
        correct = (predicted == labels).sum()
        return correct
