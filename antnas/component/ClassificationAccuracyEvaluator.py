# -*- coding: UTF-8 -*-
# @Time    : 2019-08-19 19:27
# @File    : ClassificationAccuracyEvaluator.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antnas.component.AccuracyEvaluator import *
import torch


class ClassificationAccuracyEvaluator(AccuracyEvaluator):
    def __init__(self, topk=(1,), **kwargs):
        super(ClassificationAccuracyEvaluator, self).__init__(**kwargs)
        self.total = 0
        self.correct = 0
        self.topk = topk
        self.accuracy_sum = [0 for _ in range(len(topk))]
        self.accuracy_count = [0 for _ in range(len(topk))]

    def task_type(self):
        return "CLASSIFICATION"

    def preprocess(self, *args, **kwargs):
        return args

    def caculate(self, pred, label, ignore=None):
        with torch.no_grad():
            maxk = max(self.topk)
            batch_size = label.size(0)

            _, pred = pred.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(label.view(1, -1).expand_as(pred))

            res = []
            for index, k in enumerate(self.topk):
                correct_k = correct[:k].view(-1).float().sum(0)
                res.append(correct_k.mul_(1.0 / batch_size))

                self.accuracy_sum[index] += res[index] * batch_size
                self.accuracy_count[index] += batch_size
            return res

    def _caculate_in_thread(self, *args, **kwargs):
        pass

    def accuracy(self):
        return [(float)(self.accuracy_sum[index]/self.accuracy_count[index])
                    for index in range(len(self.topk))]