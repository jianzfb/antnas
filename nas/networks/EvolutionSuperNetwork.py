# -*- coding: UTF-8 -*-
# @Time    : 2019-09-29 11:14
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
from nas.networks.SuperNetwork import SuperNetwork
from nas.interfaces.NetworkBlock import *
from nas.interfaces.NetworkCell import *
from nas.interfaces.PathRecorder import PathRecorder
import copy
import networkx as nx
import threading
from nas.networks.mutation import *
from nas.networks.crossover import *
from nas.interfaces.NetworkBlock import *
from nas.networks.nsga2 import *
population_lock = threading.Lock()


class ModelProblem(Problem):
  def __init__(self, goal='MAXIMIZE'):
    super(ModelProblem, self).__init__()
    self.max_objectives = [None, None]
    self.min_objectives = [None, None]
    self.goal = goal

  def generateIndividual(self):
    individual = Individual()
    individual.features = []
    individual.dominates = functools.partial(self.__dominates, individual1=individual)
    individual.objectives = [None, None]
    return individual

  def calculateObjectives(self, individual):
    individual.objectives[0] = self.__f1(individual)
    individual.objectives[1] = self.__f2(individual)
    for i in range(2):
      if self.min_objectives[i] is None or individual.objectives[i] < self.min_objectives[i]:
        self.min_objectives[i] = individual.objectives[i]
      if self.max_objectives[i] is None or individual.objectives[i] > self.max_objectives[i]:
        self.max_objectives[i] = individual.objectives[i]

  def __dominates(self, individual2, individual1):
    if self.goal == 'MAXIMIZE':
      worse_than_other = self.__f1(individual1) >= self.__f1(individual2) and self.__f2(individual1) >= self.__f2(
        individual2)
      better_than_other = self.__f1(individual1) > self.__f1(individual2) or self.__f2(individual1) > self.__f2(
        individual2)
      return worse_than_other and better_than_other
    else:
      worse_than_other = self.__f1(individual1) <= self.__f1(individual2) and self.__f2(individual1) <= self.__f2(
        individual2)
      better_than_other = self.__f1(individual1) < self.__f1(individual2) or self.__f2(individual1) < self.__f2(
        individual2)
      return worse_than_other and better_than_other

  def __f1(self, m):
    # model performance
    return m.objectives[0]

  def __f2(self, m):
    # model flops
    return 1.0/m.objectives[1]


class EvolutionSuperNetwork(SuperNetwork):
    def __init__(self, deter_eval, *args, **kwargs):
        super(EvolutionSuperNetwork, self).__init__(*args, **kwargs)
        self.nodes_param = None

        self.max_generation = 100
        self.current_genration = 0

        self.epoch_num_every_generation = 5
        self.current_population = None
        self.population_size = 100
        self.update_population_flag = True

        # build nsga2 evolution algorithm
        mutation_control = EvolutionMutation(multi_points=-1,
                                             max_generation=self.max_generation,
                                             k0=1.0,
                                             k1=0.8,
                                             method='based_matrices',
                                             adaptive=True)
        crossover_control = EvolutionCrossover(multi_points=10,
                                               max_generation=self.max_generation,
                                               k0=1.0,
                                               k1=0.8,
                                               method='based_matrices')
        self.evolution_control = Nsga2(ModelProblem('MAXIMIZE'),
                                       mutation_control,
                                       crossover_control)

    def forward(self, *input):
        assert len(input) == 2

        # 0.step copy network graph
        running_graph = copy.deepcopy(self.net)

        # 1.step set sampling and active on batch
        sampling = torch.Tensor()
        active = torch.Tensor()

        # 2.step parse x,y - (data,label)
        x, y = input
        input = [x]

        # 3.step get sampling architecture of
        batched_sampling, batched_archs_index = self._sample_archs(input[0].size(0), x.device)

        # 4.step forward network
        # 4.1.step set the input of network graph
        running_graph.node[self.in_node]['input'] = [*input]
        model_out = None
        node_regulizer_loss = []
        for node in self.traversal_order:
            cur_node = running_graph.node[node]
            input = self.format_input(cur_node['input'])

            if len(input) == 0:
                raise RuntimeError('Node {} has no inputs'.format(node))

            batch_size = input[0].size(0) if type(input) == list else input.size(0)
            node_sampling = self.get_node_sampling(node, batch_size, batched_sampling)
            # notify path recorder to add sampling
            sampling, active = self.add_sampling(node, node_sampling, sampling, active, self.blocks[cur_node['module']].switch)

            # 4.2.step set node sampling
            self.blocks[cur_node['module']].set_sampling(node_sampling)

            # 4.3.step set node last sampling
            if self.last_sampling is not None:
                node_last_sampling = self.last_sampling[self.path_recorder.node_index[node]]
                self.blocks[cur_node['module']].set_last_sampling(node_last_sampling)

            # 4.4.step execute node op
            out = self.blocks[cur_node['module']](input)

            # 4.5.step add regularizer loss
            if self.blocks[cur_node['module']].get_node_regularizer() is not None:
                node_regulizer_loss.append(self.blocks[cur_node['module']].get_node_regularizer())

            if node == self.out_node:
                model_out = out
                break

            # 4.6.step set successor input
            for succ in running_graph.successors(node):
                if 'input' not in running_graph.node[succ]:
                    running_graph.node[succ]['input'] = []
                running_graph.node[succ]['input'].append(out)

        # 5.step notify path recorder to update global statistics
        self.update(sampling, active)

        # 6.step compute model loss
        indiv_loss = self.loss(model_out, y)

        # 7.step compute model accuracy
        model_accuracy = self.accuray(model_out, y)

        # 8.step compute architecture loss
        optim_cost = None
        sampled_cost = None
        pruned_cost = None
        for cost, cost_eval in self.architecture_cost_evaluators.items():
            sampled_cost, pruned_cost = cost_eval.get_costs(self.architecture(sampling, active), running_graph)

            if cost == self.architecture_cost_optimization:
                optim_cost = sampled_cost

        # 更新个体信息
        population_lock.acquire()
        for arch_index, sample_index in zip(batched_archs_index, list(range(batch_size))):
            selected_count = self.current_population.population[arch_index].selected_count
            pre_accuracy = self.current_population.population[arch_index].objectives[0]
            self.current_population.population[arch_index].objectives[0] = (pre_accuracy * (selected_count-1) + model_accuracy[sample_index].item())/float(selected_count)
            self.current_population.population[arch_index].objectives[1] = sampled_cost[sample_index].item()
            self.evolution_control.problem.calculateObjectives(self.current_population.population[arch_index])
        population_lock.release()

        # 9.step compute regularizer loss
        regularizer_loss = 0.0
        if len(node_regulizer_loss) > 0 and self.kwargs['regularizer']:
            regularizer_loss = torch.Tensor(node_regulizer_loss).mean()

        # 10.step total loss
        loss = indiv_loss.mean() + 0.001 * regularizer_loss

        sampled_cost_ = torch.as_tensor(sampled_cost, device=loss.device)
        pruned_cost_ = torch.as_tensor(pruned_cost, device=loss.device)
        return loss, model_accuracy, sampled_cost_, pruned_cost_

    def save_architecture(self, path=None):
        # save current population
        for individual in self.current_population.population:
            batched_sampling = torch.Tensor(individual.features).view(1, len(individual.features))
            graph = copy.deepcopy(self.net)

            # 1.step prune sampling network
            sampling = torch.Tensor()
            active = torch.Tensor()

            for node in self.traversal_order:
                cur_node = graph.node[node]
                node_sampling = self.get_node_sampling(node, 1, batched_sampling)

                # notify path recorder to add sampling
                sampling, active = self.add_sampling(node, node_sampling, sampling, active, self.blocks[cur_node['module']].switch)

            _, pruned_architecture = self.architecture(sampling, active)

            # 2.step write to graph
            for node in self.traversal_order:
                node_sampling_val = torch.squeeze(pruned_architecture[self.path_recorder.node_index[node]]).item()
                graph.node[node]['sampled'] = int(node_sampling_val)

            # 3.step save architecture
            architecture_path = '%s_epoch_%d_accuray_%0.2f_flops_%0.15f.architecture'%(path, self.epoch, individual.objectives[0], 1.0/individual.objectives[1])
            nx.write_gpickle(graph, architecture_path)

    def sampling_param_generator(self, node_name):
        if not (node_name.startswith('CELL') or node_name.startswith('T')):
            # 不可学习，处于永远激活状态
            param_value = [0] + [1000000000000000] + [0] * (NetworkBlock.state_num - 2)
            trainable = False
        else:
            param_value = [1.0/NetworkBlock.state_num]*NetworkBlock.state_num
            trainable = False

        return nn.Parameter(torch.Tensor(param_value), requires_grad=trainable)

    @property
    def n_layers(self):
        return sum([mod.n_layers for mod in self.blocks])

    @property
    def n_comp_steps(self):
        return sum([mod.n_comp_steps for mod in self.blocks])

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

    def _sample_archs(self, batch_size, device):
        batch_size = int(batch_size)

        # 决定是否需要启动产生新种群
        mini_population = []
        population_lock.acquire()
        if self.epoch % self.epoch_num_every_generation == 0 and self.update_population_flag:
            if self.current_population is None:
                # 随机产生初代种群
                self.current_genration = 0
                self.current_population = Population()
                for index in range(self.population_size):
                    me = self.evolution_control.problem.generateIndividual()
                    me.id = index
                    me.features = self.randomSamplingArchitecture()
                    me.objectives[0] = 0
                    me.objectives[1] = 0
                    me.is_selected = False
                    me.selected_count = 0
                    self.current_population.population.append(me)
            else:
                # 产生下一代精英种群
                self.current_genration = self.epoch // self.epoch_num_every_generation
                self.evolution_control.mutation_controler.generation = self.current_genration
                self.evolution_control.crossover_controler.generation = self.current_genration

                self.current_population = self.evolution_control.evolve(self.current_population,
                                                                        self.net,
                                                                        self.blocks,
                                                                        self.architecture_node_index,
                                                                        self.architecture_cost_evaluators[self.architecture_cost_optimization].get_cost,
                                                                        device=device)
                # 重新对下一代精英种群初始化
                for individual in self.current_population.population:
                    individual.is_selected = False
                    individual.selected_count = 0
                    individual.objectives[0] = 0
                    individual.objectives[1] = 0

            self.update_population_flag = False

        # next epoch, whether need to update population
        if (self.epoch + 1) % self.epoch_num_every_generation == 0:
            self.update_population_flag = True

        # 1.step 从未被选中的个体中挑选
        candidate_individual_index = [individual.id for individual in self.current_population.population if not individual.is_selected]
        if len(candidate_individual_index) > batch_size:
            candidate_individual_index = np.random.choice(candidate_individual_index,
                                                          batch_size,
                                                          replace=False).flatten().tolist()

        for individual_index in candidate_individual_index:
            self.current_population.population[individual_index].is_selected = True
            self.current_population.population[individual_index].selected_count += 1
            mini_population.append(self.current_population.population[individual_index])

        # 2.step 随机挑选
        if len(mini_population) < batch_size:
            mini_population.extend(np.random.choice(self.current_population.population,
                                                    batch_size - len(mini_population),
                                                    replace=False).flatten().tolist())
            for individual in mini_population:
                individual.selected_count += 1

        population_lock.release()

        mini_batched_archs_index = [individual.id for individual in mini_population]
        mini_batched_sampling = [individual.features for individual in mini_population]
        mini_batched_sampling = torch.as_tensor(mini_batched_sampling, device=device)
        return mini_batched_sampling, mini_batched_archs_index

    def randomSamplingArchitecture(self):
        feature = [None for _ in range(len(self.traversal_order))]
        for node_name in self.traversal_order:
            cur_node = self.net.node[node_name]
            if not (node_name.startswith('CELL') or node_name.startswith('T')):
                # 不可学习，处于永远激活状态
                feature[cur_node['sampling_param']] = int(1)
            else:
                if self.blocks[cur_node['module']].switch:
                    feature[cur_node['sampling_param']] = int(np.random.randint(0, 2))
                else:
                    feature[cur_node['sampling_param']] = int(np.random.randint(0, NetworkBlock.state_num))
        return feature

    def __str__(self):
        model_descr = 'Model:{}\n\t{} nodes\n\t{} blocks\n\t{} parametrized layers\n\t{} computation steps\n\t{} parameters\n\t{} meta-params'
        return model_descr.format(type(self).__name__,
                                  self.graph.number_of_nodes(),
                                  len(self.blocks),
                                  self.n_layers,
                                  self.n_comp_steps,
                                  sum(i.numel() for i in self.parameters()),
                                  len(self.sampling_parameters))