# -*- coding: UTF-8 -*-
# @Time    : 2020-04-05 22:19
# @File    : EvolutionSuperNetwork.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from nas.networks.SuperNetwork import *
from nas.component.NetworkBlock import *
from nas.component.NetworkCell import *
from nas.component.PathRecorder import PathRecorder
from nas.component.NetworkBlock import *
import copy
import networkx as nx
import threading
from nas.networks.nsga2 import *
from nas.networks.bayesian import *
from nas.networks.mutation import *
from nas.networks.crossover import *


class EvolutionSuperNetwork(SuperNetwork):
    def __init__(self, *args, **kwargs):
        super(EvolutionSuperNetwork, self).__init__(*args, **kwargs)
        self.warmup_epochs = kwargs.get("warmup_epochs", 10)
        self.evo_epochs = kwargs.get("evo_epochs", 10)
        self.population_size = kwargs.get("population_size", 100)
        self.population = None

    def search_init(self, *args, **kwargs):
        self.population = Population()
        self.population.current_genration = 0
        for index in range(self.population_size):
            individual = self.problem.generateIndividual()
            feature = [None for _ in range(len(self.traversal_order))]
            for node_name in self.traversal_order:
                cur_node = self.net.node[node_name]
                if not (node_name.startswith('CELL') or node_name.startswith('T')):
                    # 不可学习，处于永远激活状态
                    feature[cur_node['sampling_param']] = int(1)
                else:
                    if not self.blocks[cur_node['module']].structure_fixed:
                        feature[cur_node['sampling_param']] = int(np.random.randint(0, 2))
                    else:
                        feature[cur_node['sampling_param']] = int(np.random.randint(0, NetworkBlock.state_num))
            individual.features = feature
            individual.accuracy = 0.0
            self.problem.calculateObjectives(individual)
            self.population.population.append(individual)

    def search(self, *args, **kwargs):
        # NSGAII configure
        crossover_multi_points = kwargs.get('corssover_multi_points', 5)
        mutation_multi_points = kwargs.get('mutation_multi_points', -1)

        # build work folder
        epoch = kwargs.get('epoch')
        folder = kwargs.get('folder', './supernetwork/')
        epoch_folder = os.path.join(folder, 'evo_epoch_%d'%kwargs.get('epoch'))
        if not os.path.exists(epoch_folder):
            os.makedirs(epoch_folder)

        # NSGAII evolution algorithm
        mutation_control = EvolutionMutation(multi_points=mutation_multi_points,
                                             max_generation=self.evo_epochs,
                                             k0=1.0,
                                             k1=1.5,
                                             method='based_matrices',
                                             adaptive=True)
        crossover_control = EvolutionCrossover(multi_points=crossover_multi_points,
                                               max_generation=self.evo_epochs,
                                               k0=1.0,
                                               k1=0.8,
                                               method='based_matrices',
                                               size=self.population_size)

        nsga = \
            Nsga2(self.problem,
                  mutation_control,
                  crossover_control,
                  num_of_generations=1,
                  callback=functools.partial(self._evolution_callback_func,
                                             arc_loss=self.problem.arc_loss,
                                             folder=epoch_folder))

        explore_position = []
        for node_name in self.traversal_order:
            cur_node = self.net.node[node_name]
            if node_name.startswith('CELL') or node_name.startswith('T'):
                explore_position.append(cur_node['sampling_param'])

        self.population = nsga.evolve(self.population,
                                      graph=self.net,
                                      blocks=self.blocks,
                                      explore_position=explore_position)

        # save architecture
        for individual in self.population:
            batched_sampling = torch.Tensor(individual.features).view(1, len(individual.features))

            # 3.1.step prune sampling network
            sampling = torch.Tensor()
            active = torch.Tensor()

            for node in self.traversal_order:
                cur_node = self.net.node[node]
                node_sampling = self.get_node_sampling(node, 1, batched_sampling)

                # notify path recorder to add sampling
                sampling, active = self.path_recorder.add_sampling(node,
                                                                   node_sampling,
                                                                   sampling,
                                                                   active,
                                                                   self.blocks[cur_node['module']].structure_fixed)

            _, pruned_arch = self.path_recorder.get_arch(self.out_node, sampling, active)

            # 3.2.step write to graph
            for node in self.traversal_order:
                node_sampling_val = torch.squeeze(pruned_arch[self.path_recorder.node_index[node]]).item()
                self.net.node[node]['sampled'] = int(node_sampling_val)

            # 3.3.step get architecture parameter number
            parameter_num = 0
            for node in self.traversal_order:
                sampled_state = self.net.node[node]['sampled']
                cur_node = self.net.node[node]
                parameter_num += self.blocks[cur_node['module']].get_param_num(None)[sampled_state]

            # 3.4.step save architecture
            architecture_info = ''
            if self.problem.arc_loss == 'comp':
                architecture_info = 'flops_%d' % int(individual.objectives[1])
            elif self.problem.arc_loss == 'latency':
                architecture_info = 'latency_%0.2f' % individual.objectives[1]
            else:
                architecture_info = 'para_%d' % int(individual.objectives[1])

            architecture_tag = \
                'evo_%d_accuray_%0.4f_%s_params_%d.architecture' % (epoch,
                                                                    1.0-individual.objectives[0],
                                                                    architecture_info,
                                                                    int(parameter_num))
            architecture_path = os.path.join(epoch_folder, architecture_tag)
            nx.write_gpickle(self.net, architecture_path)

    def forward(self, x, y, arc=None, epoch=None, warmup=False):
        # 1.step parse x,y - (data,label)
        input = [x]

        batched_sampling = None
        if arc is None:
            batched_sampling = self.sample_arch(warmup=warmup)
        else:
            batched_sampling = arc

        # 3.step forward network
        # 3.1.step set the input of network graph
        data_dict = {}
        data_dict[self.in_node] = [*input]

        model_out = None
        for node in self.traversal_order:
            cur_node = self.net.node[node]
            input = self.format_input(data_dict[node])

            if len(input) == 0:
                raise RuntimeError('Node {} has no inputs'.format(node))

            batch_size = input[0].size(0) if type(input) == list else input.size(0)
            node_sampling = self.get_node_sampling(node, batch_size, batched_sampling)
            # notify path recorder to add sampling
            # sampling, active = self.add_sampling(node, node_sampling, sampling, active, self.blocks[cur_node['module']].switch)

            # 3.2.step execute node op
            out = self.blocks[cur_node['module']](input, node_sampling)

            if node == self.out_node:
                model_out = out
                break

            # 3.3.step set successor input
            for succ in self.graph.successors(node):
                if succ not in data_dict:
                    data_dict[succ] = []

                data_dict[succ].append(out)

        # 4.step compute model loss
        indiv_loss = self.loss(model_out, y)

        # 5.step compute model accuracy
        model_accuracy = self.accuray(model_out, y)

        # 6.step total loss
        loss = indiv_loss.mean()
        return loss, model_accuracy, None, None

    def sample_arch(self, *args, **kwargs):
        warmup = kwargs['warmup']
        batch_size = kwargs['batch_size']
        batch_arch_list = []
        if warmup:
            # uniform sampling
            while len(batch_arch_list) < batch_size:
                feature = [None for _ in range(len(self.traversal_order))]
                for node_name in self.traversal_order:
                    cur_node = self.net.node[node_name]
                    if not (node_name.startswith('CELL') or node_name.startswith('T')):
                        # 不可学习，处于永远激活状态
                        feature[cur_node['sampling_param']] = int(1)
                    else:
                        if not self.blocks[cur_node['module']].structure_fixed:
                            feature[cur_node['sampling_param']] = int(np.random.randint(0, 2))
                        else:
                            feature[cur_node['sampling_param']] = int(np.random.randint(0, NetworkBlock.state_num))

                batch_arch_list.append(feature)

            batch_arch = torch.as_tensor(batch_arch_list)
            return batch_arch
        else:
            # sampling from population
            individual_index_list = [individual_index for individual_index, individual in enumerate(self.population.population)]
            sampling_arch_list = np.random.choice(individual_index_list,
                                                  batch_size).flatten().tolist()
            for index in range(batch_size):
                batch_arch_list.append(self.population.population[sampling_arch_list[index]].features)

            batch_arch = torch.as_tensor(batch_arch_list)
            return batch_arch

    def get_node_sampling(self, node_name, batch_size, batched_sampling):
        """
        Get a batch of sampling for the given node on the given output.
        Fires a "sampling" event with the node name and the sampling Variable.
        :param node_name: Name of the node to sample
        :param out: Tensor on which the sampling will be applied
        :return: A Variable brodcastable to out's size, with all dimensions equals to one except the first one (batch)
        """

        sampling_dim = [batch_size] + [1] * 3
        node = self.net.node[node_name]
        node_sampling = batched_sampling[:, node['sampling_param']].contiguous().view(sampling_dim)

        return node_sampling