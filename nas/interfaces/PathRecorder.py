import abc

import networkx as nx
import torch
from torch.autograd import Variable
from nas.utils.globalval import *


class PathRecorder(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, graph, default_out=None):
        self.graph = graph
        self.default_out = default_out
        self.n_nodes = self.graph.number_of_nodes()

        # create node-to-index and index-to-node mapping
        self.node_index = {}
        self.rev_node_index = [None] * self.n_nodes
        for i, node in enumerate(nx.topological_sort(self.graph)):  # To have index ordered as traversal_order
            self.node_index[node] = i
            self.rev_node_index[i] = node

        # update by multi threading
        self.global_sampling = None
        self.median_sampling = None
        self.n_samplings = 0

    def update_global_sampling(self, used_nodes):
        global lock
        lock.acquire()
        self.n_samplings += 1
        mean_sampling = used_nodes.mean(1).squeeze()

        if self.global_sampling is None:
            self.global_sampling = mean_sampling
        else:
            self.global_sampling += (1 / self.n_samplings) * (mean_sampling - self.global_sampling)

        if self.median_sampling is None:
            self.median_sampling = [used_nodes]
        else:
            self.median_sampling.append(used_nodes)
        lock.release()

    def update(self, sampling, active):
        if self.default_out is not None and active is not None and active.numel() > 0:
            # pruned = self.get_pruned_architecture(self.default_out, sampling, active)
            self.update_global_sampling(sampling)

    def get_and_reset(self):
        multi_cards_sampling = torch.cat(self.median_sampling,-1)
        median_sampling = multi_cards_sampling.median(1).values
        self.median_sampling = None
        return median_sampling

    def add_sampling(self, node_name, node_sampling, sampling, active, switch):
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

        if not switch:
            # 对于非开关节点，不考虑0状态
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

    # def get_graph_paths(self, out_node):
    #     sampled, pruned = self.get_architectures(out_node)
    #
    #     real_paths = []
    #     for i in range(pruned.size(1)):  # for each batch element
    #         path = [self.rev_node_index[ind] for ind, used in enumerate(pruned[:, i]) if used == 1]
    #         real_paths.append(path)
    #
    #     res = self.get_used_nodes(pruned.t())
    #
    #     assert real_paths == res
    #
    #     sampling_paths = []
    #     for i in range(sampled.size(1)):  # for each batch element
    #         path = dict((self.rev_node_index[ind], elt) for ind, elt in enumerate(sampled[:, i]))
    #         sampling_paths.append(path)
    #
    #     self.update_global_sampling(pruned)
    #
    #     return real_paths, sampling_paths

    def get_posterior_weights(self):
        return dict((self.rev_node_index[ind], elt) for ind, elt in enumerate(self.global_sampling))

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

    def get_architectures(self, out_node, sampling, active):
        return self.get_sampled_architecture(sampling, active), self.get_pruned_architecture(out_node, sampling,active)

    def get_sampled_architecture(self, sampling, active):
        return sampling

    def get_pruned_architecture(self, out_node, sampling, active):
        return active[self.node_index[out_node]] * sampling

    def get_state(self):
        return {'node_index': self.node_index,
                'rev_node_index': self.rev_node_index,
                'global_sampling': self.global_sampling,
                'n_samplings': self.n_samplings}

    def load_state(self, state):
        for key, val in state.items():
            assert hasattr(self, key)
            setattr(self, key, val)
