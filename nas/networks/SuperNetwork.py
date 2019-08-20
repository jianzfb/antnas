from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function


import networkx as nx
from torch import nn
import torch


class SuperNetwork(nn.Module):
    def __init__(self):
        super(SuperNetwork, self).__init__()

        self.net = None
        self.traversal_order = None
        self.in_node = None
        self.out_node = None
        self.observer = None

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
        output = []
        self.net.node[self.in_node]['input'] = [*input]
        # self.net.node[self.in_node]['input'] = input

        for node in self.traversal_order:
            cur_node = self.net.node[node]
            input = self.format_input(cur_node['input'])
            out = cur_node['module'](input)
            cur_node['input'] = []

            if node == self.out_node:
                output.append(out)

            for succ in self.net.successors_iter(node):
                if 'input' not in self.net.node[succ]:
                    self.net.node[succ]['input'] = []
                self.net.node[succ]['input'].append(out)

        return output[0]

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