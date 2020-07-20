# -*- coding: UTF-8 -*-
# @Time    : 2019-08-19 19:25
# @File    : AccuracyEvaluator.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import abc


class AccuracyEvaluator(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, *args, **kwargs):
        super(AccuracyEvaluator, self).__init__()

    @abc.abstractmethod
    def task_type(self):
        raise NotImplementedError

    @abc.abstractmethod
    def accuracy(self, **kwargs):
        raise NotImplementedError