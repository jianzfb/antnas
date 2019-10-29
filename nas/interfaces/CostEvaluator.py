# -*- coding: UTF-8 -*-
# @Time    : 2019-08-19 18:16
# @File    : CostEvaluator.py
# @Author  : jian<jian@mltalker.com>
import abc

class CostEvaluator(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, *args, **kwargs):
        super(CostEvaluator, self).__init__()
        self.model = kwargs.get('model')
        self.main_cost = kwargs.get('main_cost')
        self.kwargs = kwargs
        self.costs = None

    @abc.abstractmethod
    def get_cost(self, **kwargs):
        raise NotImplementedError

    def get_costs(self, architectures, graph):
        return [self.get_cost(arch, graph) for arch in architectures]

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
