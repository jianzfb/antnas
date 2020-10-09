# -*- coding: UTF-8 -*-
# @Time    : 2019-08-19 18:16
# @File    : CostEvaluator.py
# @Author  : jian<jian@mltalker.com>
import abc
import numpy as np


class CostEvaluator(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, *args, **kwargs):
        super(CostEvaluator, self).__init__()
        self.model = kwargs.get('model')
        self.main_cost = kwargs.get('main_cost')
        self.input_node = kwargs.get('input_node', None)
        self.input_shape = kwargs.get('input_shape', None)
        self.mode = kwargs.get('mode', 'default')   # default/heterogeneous(算子+设备)
        self.transfer_cost = kwargs.get('transfer_cost', None)
        self.kwargs = kwargs
        self.costs = None

    @abc.abstractmethod
    def get_cost(self, *args, **kwargs):
        raise NotImplementedError

    def get_costs(self, architectures, device=None):
        return [self.get_cost(arch, device) for arch in architectures]

    def init_costs(self, *args, **kwargs):
        pass

    def get_state(self):
        return {}

    def load_state(self, state):
        for key, val in state.items():
            assert hasattr(self, key)
            setattr(self, key, val)

    def new_epoch(self):
        pass
