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
    def __init__(self, *args, **kwargs):
        super(ClassificationAccuracyEvaluator, self).__init__(*args, **kwargs)
        self.total = 0
        self.correct = 0

    def task_type(self):
        return "CLASSIFICATION"

    def preprocess(self, *args,**kwargs):
        return args

    def caculate(self, pred, label, ignore=None):
        pred_cpy = torch.Tensor()
        label_cpy = torch.Tensor()
        pred_cpy.resize_(pred.size()).copy_(pred)
        label_cpy.resize_(label.size()).copy_(label)
        AccuracyEvaluator.process_queue.put((self, (pred_cpy, label_cpy)))

    def _caculate_in_thread(self, *args, **kwargs):
        pred, label = args
        _, predicted = torch.max(pred.data, 1)
        correct = torch.sum((predicted == label).float())

        lock = kwargs['lock']
        lock.acquire()
        self.correct += correct
        self.total += label.shape[0]
        lock.release()

    def accuracy(self):
        return float(self.correct / self.total)