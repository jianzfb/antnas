from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from nas.networks.SuperNetwork import SuperNetwork
from nas.interfaces.NetworkBlock import *
from nas.interfaces.PathRecorder import PathRecorder
import copy
import networkx as nx


class StochasticSuperNetwork(SuperNetwork):
    INIT_NODE_PARAM = 3

    def __init__(self, deter_eval, *args, **kwargs):
        super(StochasticSuperNetwork, self).__init__(*args, **kwargs)
        self.nodes_param = None
        self.probas = None
        self.entropies = None
        self.mean_entropy = None
        self.deter_eval = deter_eval

        self.batched_sampling = None
        self.batched_log_probas = None

    def get_sampling(self, node_name, batch_size):
        """
        Get a batch of sampling for the given node on the given output.
        Fires a "sampling" event with the node name and the sampling Variable.
        :param node_name: Name of the node to sample
        :param out: Tensor on which the sampling will be applied
        :return: A Variable brodcastable to out's size, with all dimensions equals to one except the first one (batch)
        """

        sampling_dim = [batch_size] + [1] * 3

        static_sampling = self._get_node_static_sampling(node_name)
        if static_sampling is not None:
            sampling = torch.Tensor().resize_(*sampling_dim).fill_(static_sampling)
            sampling = Variable(sampling, requires_grad=False)
        else:
            node = self.net.node[node_name]
            sampling = self.batched_sampling[:, node['sampling_param']].contiguous().view(sampling_dim)

        self.fire(type='sampling', node=node_name, value=sampling)
        return sampling

    def _get_node_static_sampling(self, node_name):
        """
        Method used to check if the sampling should be done or if the node is static.
        Raise error when used with an old version of the graph.
        :param node_name:
        :return: The value of the sampling (0 or 1) if the given node is static. None otherwise.

        """
        node = self.net.node[node_name]
        if 'sampling_val' in node or 'sampling_param' not in node or node['sampling_param'] is None:
            raise RuntimeError('Deprecated method. {} node has no attribute `sampling_param` or has attribute '
                               '`sampling_val`.All nodes should now have a param '
                               '(static ones are np.inf with non trainable param).'.format(node_name))

        if not self.training and self.deter_eval:
            # Model is in deterministic evaluation mode
            if not isinstance(node['sampling_param'], int):
                return (F.sigmoid(self.sampling_parameters[node['sampling_param']]) > 0.5).item()

        return None

    def forward(self, *input):
        # 0.step build auxiliary running info
        if self.running_path_recorder is None:
            self.running_path_recorder = PathRecorder(self.graph, self.out_node)
            self.subscribe(self.running_path_recorder.new_event)

        running_graph = copy.deepcopy(self.net)

        # 1.step start new iteration
        self.fire(type='new_iteration')
        assert len(input) == 2

        x, y = input
        input = [x]

        # 2.step sampling network
        self._sample_archs(input[0].size(0))
        running_graph.node[self.in_node]['input'] = [*input]

        # 3.step forward sampling network
        model_out = None
        for node in self.traversal_order:
            cur_node = running_graph.node[node]
            input = self.format_input(cur_node.pop('input'))

            if len(input) == 0:
                raise RuntimeError('Node {} has no inputs'.format(node))
            batch_size = input[0].size(0) if type(input) == list else input.size(0)
            sampling = self.get_sampling(node, batch_size)
            self.net.node[node]['sampled'] = torch.squeeze(sampling)[0].item()
            self.blocks[cur_node['module']].sampling(sampling)
            out = self.blocks[cur_node['module']](input)

            if node == self.out_node:
                model_out = out
                break

            for succ in running_graph.successors(node):
                if 'input' not in running_graph.node[succ]:
                    running_graph.node[succ]['input'] = []
                running_graph.node[succ]['input'].append(out)

        # 4.step compute model loss
        indiv_loss = self.loss(model_out, y)

        # 5.step compute model accuracy
        model_accuracy = self.accuray(model_out, y)

        # 6.step compute architecture loss
        optim_cost = None
        sampled_cost = None
        pruned_cost = None
        for cost, cost_eval in self.architecture_cost_evaluators.items():
            sampled_cost, pruned_cost = cost_eval.get_costs(self.architecture)

            if cost == self.architecture_cost_optimization:
                optim_cost = sampled_cost

        cost = (optim_cost + self.architecture_penalty * self.architecture_consistence) - self.architecture_objective_cost

        if self.architecture_objective_method == 'max':
            cost.clamp_(min=0)
        elif self.architecture_objective_method == 'abs':
            cost.abs_()

        cost = indiv_loss.data.new(cost.size()).copy_(cost)
        rewards = -(indiv_loss.data.squeeze() + self.architecture_lambda * cost)
        mean_reward = rewards.mean()
        rewards = (rewards - mean_reward)

        rewards = Variable(rewards)
        sampling_loss = self.architecture_loss(rewards=rewards)
        loss = indiv_loss.mean() + sampling_loss

        # sampled_cost = sampled_cost.mean()
        # pruned_cost = pruned_cost.mean()
        nx.write_gpickle(self.net, "test.gpickle")
        return loss, model_accuracy

    def _sample_archs(self, batch_size):
        batch_size = int(batch_size)

        # params = torch.stack([p for p in self.sampling_parameters], dim=1)
        # probas_resized = params.sigmoid().expand(batch_size, len(self.sampling_parameters))
        # distrib = torch.distributions.Bernoulli(probas_resized)
        #
        key_map = {}
        for i, node in enumerate(self.graph.nodes()):
            key_map[node] = i

        params = torch.stack([self.sampling_parameters[key_map[order_name]] for order_name in self.traversal_order], dim=0)
        probas_resized = params.softmax(dim=-1).expand(batch_size,
                                                       params.size(0),
                                                       params.size(1))
        distrib = torch.distributions.categorical.Categorical(probas_resized)

        self.batched_sampling = distrib.sample()
        self.batched_log_probas = distrib.log_prob(self.batched_sampling)

    @property
    def n_layers(self):
        return sum([mod.n_layers for mod in self.blocks])

    @property
    def n_comp_steps(self):
        return sum([mod.n_comp_steps for mod in self.blocks])

    def reinit_sampling_params(self):
        new_params = nn.ParameterList()
        for p in self.sampling_parameters:
            if p.requires_grad:
                param_value = self.INIT_NODE_PARAM
            else:
                param_value = p.data[0]
            new_params.append(nn.Parameter(p.data.new(([param_value])), requires_grad=p.requires_grad))

        self.sampling_parameters = new_params

    def update_probas_and_entropies(self):
        if self.nodes_param is None:
            self._init_nodes_param()

        self.probas = {}
        self.entropies = {}
        self.mean_entropy = .0
        for node, props in self.graph.node.items():
            param = self.sampling_parameters[props['sampling_param']]
            p = param.sigmoid().item()
            self.probas[node] = p
            if p in [0, 1]:
                e = 0
            else:
                e = -(p * np.log2(p)) - ((1 - p) * np.log2(1 - p))
            self.entropies[node] = e
            self.mean_entropy += e
        self.mean_entropy /= self.graph.number_of_nodes()

    def _init_nodes_param(self):
        self.nodes_param = {}
        for node, props in self.graph.node.items():
            if 'sampling_param' in props and props['sampling_param'] is not None:
                self.nodes_param[node] = props['sampling_param']

    def architecture_loss(self, *args, **kwargs):
        rewards = kwargs['rewards']
        sampling_loss_val = (-self.batched_log_probas * rewards.unsqueeze(dim=1)).sum()
        return sampling_loss_val

    def __str__(self):
        model_descr = 'Model:{}\n\t{} nodes\n\t{} blocks\n\t{} parametrized layers\n\t{} computation steps\n\t{} parameters\n\t{} meta-params'
        return model_descr.format(type(self).__name__,
                                  self.graph.number_of_nodes(),
                                  len(self.blocks),
                                  self.n_layers,
                                  self.n_comp_steps,
                                  sum(i.numel() for i in self.parameters()),
                                  len(self.sampling_parameters))