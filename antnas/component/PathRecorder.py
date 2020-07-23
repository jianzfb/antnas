import abc

import networkx as nx
import torch
import numpy as np
from antnas.component.NetworkBlock import *
from torch.autograd import Variable


class PathRecorder(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, graph, default_out=None):
        self.graph = graph
        self.default_out = default_out
        self.n_nodes = self.graph.number_of_nodes()

        # create node-to-index and index-to-node mapping
        self.node_index = {}                                # 节点名字与拓扑排序的对应关系
        self.rev_node_index = [None] * self.n_nodes
        for i, node in enumerate(nx.topological_sort(self.graph)):  # To have index ordered as traversal_order
            self.node_index[node] = i
            self.rev_node_index[i] = node                   # 拓扑排序与节点名字的对应关系

    def add_sampling(self, node_name, node_sampling, sampling, active, structure_fixed):
        node_sampling = node_sampling.float()
        if isinstance(node_sampling, torch.Tensor):
            node_sampling = node_sampling.data.cpu().squeeze()
        if active is None:
            raise RuntimeError("'new_iteration' should be called before each evaluation.")
        if node_sampling.dim() == 0:
            node_sampling.unsqueeze_(0)
        if node_sampling.dim() != 1:
            raise ValueError("'sampling' param should be of dimension one.")

        batch_size = node_sampling.size(0)

        if active.numel() == 0 and sampling.numel() == 0:
            active.resize_(self.n_nodes, self.n_nodes, batch_size).zero_()
            sampling.resize_(self.n_nodes, batch_size).zero_()

        node_ind = self.node_index[node_name]
        incoming = active[node_ind]

        # write to sampling
        sampling[self.node_index[node_name]] = node_sampling

        if structure_fixed:
            # 对于结构固定节点，任意状态均表示有效操作
            # 对于非结构固定节点，0状态表示断开操作，1状态表示有效操作
            node_sampling = node_sampling + 1

        # parse graph is active?
        if len(list(self.graph.predecessors(node_name))) == 0:
            incoming[node_ind] += node_sampling

        for prev in self.graph.predecessors(node_name):
            incoming += active[self.node_index[prev]]

        assert incoming.size() == torch.Size((self.n_nodes, batch_size))

        has_inputs = incoming.view(-1, batch_size).max(0)[0]
        has_outputs = ((has_inputs * node_sampling) != 0).float()

        incoming[node_ind] += node_sampling

        sampling_mask = has_outputs.expand(self.n_nodes, batch_size)
        incoming *= sampling_mask

        active[node_ind] = (incoming != 0).float()
        return sampling, active

    def get_used_nodes(self, architectures):
        """
        Translates each architecture from a vector representation to a list of the nodes it contains
        :param architectures: a batch or architectures in format batch_size * n_nodes
        :return: a list of batch_size elements, each elements being a list of nodes.
        """
        res = []
        for arch in architectures:
            nodes = [self.rev_node_index[idx] for idx, used in enumerate(arch) if used == 1]
            res.append(nodes)
        return res

    def get_consistence(self, node, sampling, active):
        """
        Get an indicator of consistence for each sampled architecture up to the given node in last batch.

        :param node: The target node.
        :return: a ByteTensor containing one(zero) only if the architecture is consistent and the param is True(False).
        """
        return active[self.node_index[node]].sum(0) != 0

    def is_consistent(self, model, sampling, active):
        model.eval()
        with torch.no_grad():
            input = torch.ones(1, *model.input_size)
            model(input)
        consistence = self.get_consistence(model.out_node, sampling, active)
        return consistence.sum() != 0

    def get_arch(self, out_node, sampling, active):
        return self.get_sampled_arch(sampling, active), self.get_pruned_arch(out_node, sampling, active)

    def get_sampled_arch(self, sampling, active):
        return sampling

    def get_pruned_arch(self, out_node, sampling, active):
        return active[self.node_index[out_node]] * sampling
