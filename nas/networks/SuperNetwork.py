from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function


import networkx as nx
from torch import nn
from nas.implem.ParameterCostEvaluator import ParameterCostEvaluator
from nas.implem.LatencyCostEvaluator import LatencyCostEvaluator
from nas.implem.ComputationalCostEvaluator import ComputationalCostEvaluator
import torch
from nas.interfaces.NetworkBlock import *
from nas.interfaces.PathRecorder import PathRecorder


class SuperNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SuperNetwork, self).__init__()

        self.net = None
        self.traversal_order = None
        self.in_node = None
        self.out_node = None

        self.path_recorder = None
        self._cost_evaluators = None

        self._cost_optimization = kwargs['cost_optimization']
        self._architecture_penalty = kwargs['arch_penalty']
        self._objective_cost = kwargs['objective_cost']
        self._objective_method = kwargs['objective_method']
        self._architecture_lambda = kwargs['lambda']

        cost_evaluators = {
            'comp': ComputationalCostEvaluator,
            'latency': LatencyCostEvaluator,
            'param': ParameterCostEvaluator
        }

        used_ce = {}
        for k in kwargs["cost_evaluation"]:
            used_ce[k] = cost_evaluators[k](model=self,
                                            main_cost=(k == self._cost_optimization),
                                            **kwargs)

        self._cost_evaluators = used_ce
        self._last_sampling = None

        # global configure
        self.kwargs = kwargs

        self._epoch = 0

    def set_graph(self, network, in_node, out_node):
        self.net = network
        if not nx.is_directed_acyclic_graph(self.net):
            raise ValueError('A Super Network must be defined with a directed acyclic graph')

        self.traversal_order = list(nx.topological_sort(self.net))
        self.in_node = in_node
        self.out_node = out_node

        # TODO Allow several input and/or output nodes
        if self.traversal_order[0] != in_node or self.traversal_order[-1] != out_node:
            raise ValueError('Seems like the given graph is broken')

        self.path_recorder = PathRecorder(self.net, self.out_node)

    def forward(self, *input):
        raise NotImplementedError

    @property
    def input_size(self):
        if not hasattr(self, '_input_size'):
            raise RuntimeError('SuperNetworks should have an `_input_size` attribute.')
        return self._input_size

    @staticmethod
    def format_input(input):
        if (isinstance(input, tuple) or isinstance(input, list)) and len(input) == 1:
            input = input[0]
        return input

    def architecture_loss(self, *args, **kwargs):
        raise NotImplementedError

    def architecture_optimize(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, sampling, active):
        self.path_recorder.update(sampling, active)

    def add_sampling(self, node_name, node_sampling, sampling, active, switch=False):
        return self.path_recorder.add_sampling(node_name, node_sampling, sampling, active, switch)

    def loss(self, predictions, labels):
        raise NotImplementedError

    def accuray(self, predictions, labels):
        raise NotImplementedError

    @property
    def architecture_cost_evaluators(self):
        return self._cost_evaluators

    def architecture(self, sampling, active):
        return self.path_recorder.get_architectures(self.out_node, sampling, active)

    def architecture_consistence(self, sampling, active):
        return self.path_recorder.get_consistence(self.out_node, sampling, active).float()

    @property
    def architecture_cost_optimization(self):
        return self._cost_optimization

    @property
    def architecture_penalty(self):
        return self._architecture_penalty

    @property
    def architecture_objective_cost(self):
        return self._objective_cost

    @property
    def architecture_objective_method(self):
        return self._objective_method

    @property
    def architecture_lambda(self):
        return self._architecture_lambda

    @property
    def architecture_node_index(self):
        return self.path_recorder.node_index

    def save_architecture(self, folder=None, name=None):
        pass

    def get_path_recorder(self):
        return self.path_recorder

    @property
    def last_sampling(self):
        return self._last_sampling

    @last_sampling.setter
    def last_sampling(self, val):
        self._last_sampling = val

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, val):
        self._epoch = val

    def preprocess(self):
        pass

    def afterprocess(self):
        pass

    def plot(self, path=None):
        pass