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
        AccuracyEvaluator.process_queue.put((self, (pred, label, ignore)))

    def _caculate_in_thread(self, *args, **kwargs):
        pred, label, _ = args

        _, predicted = torch.max(pred.data, 1)
        correct = torch.sum((predicted == label).float())
        self.correct += correct

        lock = kwargs['lock']
        lock.acquire()
        self.total += label.shape[0]
        lock.release()

    def accuracy(self):
        return float(self.correct / self.total)