from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function


import networkx as nx
from torch import nn
from nas.implem.ParameterCostEvaluator import ParameterCostEvaluator
from nas.implem.TimeCostEvaluator import TimeCostEvaluator
from nas.implem.ComputationalCostEvaluator import ComputationalCostEvaluator
import torch


class Event(object):
    pass


class SuperNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SuperNetwork, self).__init__()

        self.net = None
        self.traversal_order = None
        self.in_node = None
        self.out_node = None
        self.observer = None
        self.callbacks = None

        self.running_path_recorder = None
        self._cost_evaluators = None

        self._cost_optimization = None
        self._architecture_penalty = None
        self._objective_cost = None
        self._objective_method = 'max'
        self._architecture_lambda = 1
        self._cost_evaluation = []

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

    def forward(self, *input):
        # output = []
        # self.net.node[self.in_node]['input'] = [*input]
        # # self.net.node[self.in_node]['input'] = input
        #
        # for node in self.traversal_order:
        #     cur_node = self.net.node[node]
        #     input = self.format_input(cur_node['input'])
        #     out = cur_node['module'](input)
        #     cur_node['input'] = []
        #
        #     if node == self.out_node:
        #         output.append(out)
        #
        #     for succ in self.net.successors_iter(node):
        #         if 'input' not in self.net.node[succ]:
        #             self.net.node[succ]['input'] = []
        #         self.net.node[succ]['input'].append(out)
        #
        # return output[0]
        return None

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

    def consistent_path(self, sampling, graph, node_name, active):
        batch_size = sampling.size(0)
        node_num = len(self.net.node)

        node_ind = self.net.node[node_name]['sampling_param']
        incoming = active[node_ind]

        if len(list(graph.predecessors(node_name))) == 0:
            incoming[node_ind] += sampling

        for prev in graph.predecessors(node_name):
            incoming += active[self.net.node[prev]['sampling_param']]

        has_inputs = incoming.view(-1, batch_size).max(0)[0]
        has_outputs = ((has_inputs * sampling) != 0).float()

        incoming[node_ind] += sampling

        sampling_mask = has_outputs.expand(node_num, batch_size)
        incoming *= sampling_mask

        active[node_ind] = (incoming != 0).float()
        return active

    def architecture_loss(self, *args, **kwargs):
        raise NotImplementedError

    def architecture_optimize(self, *args, **kwargs):
        raise NotImplementedError

    def subscribe(self, callback):
        if self.callbacks is None:
            self.callbacks = []

        self.callbacks.append(callback)

    def fire(self, **attrs):
        e = Event()
        e.source = self
        for k, v in attrs.items():
            setattr(e, k, v)
        for fn in self.callbacks:
            fn(e)

    def loss(self, predictions, labels):
        raise NotImplementedError

    def accuray(self, predictions, labels):
        raise NotImplementedError

    @property
    def architecture_cost_evaluators(self):
        if self._cost_evaluators is not None:
            return self._cost_evaluators

        cost_evaluators = {
            'comp': ComputationalCostEvaluator,
            'time': TimeCostEvaluator,
            'param': ParameterCostEvaluator
        }

        used_ce = {}
        for k in self._cost_evaluation:
            used_ce[k] = cost_evaluators[k](path_recorder=self.running_path_recorder,
                                            model=self,
                                            main_cost=(k == self._cost_optimization))

        self._cost_evaluators = used_ce
        return self._cost_evaluators

    @property
    def architecture(self):
        return self.running_path_recorder.get_architectures(self.out_node)

    @property
    def architecture_consistence(self):
        return self.running_path_recorder.get_consistence(self.out_node).float()


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