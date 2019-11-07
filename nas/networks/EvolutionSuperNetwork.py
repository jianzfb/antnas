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
  def __init__(self, goal='MAXIMIZE', alpha=0.7, beta=0.7, threshold=0):
    super(ModelProblem, self).__init__()
    self.max_objectives = [None, None]
    self.min_objectives = [None, None]
    self.goal = goal
    self.alpha = alpha
    self.beta = beta
    self.T = threshold

  def generateIndividual(self):
    individual = Individual()
    individual.features = []
    individual.dominates = functools.partial(self.__dominates, individual1=individual)
    individual.objectives = [None, None]
    individual.values = [None, None]
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
    # model accuracy
    # from mnas
    #w = self.alpha if m.values[1] <= self.T else self.beta
    #return m.values[0] * float(np.power(m.values[1]/self.T, w))

    # minimize model error
    return 1.0 - m.values[0]

  def __f2(self, m):
    # model flops/latency
    return m.values[1]


class EvolutionSuperNetwork(SuperNetwork):
    def __init__(self, deter_eval, *args, **kwargs):
        super(EvolutionSuperNetwork, self).__init__(*args, **kwargs)
        self.nodes_param = None

        self.max_generation = 100
        self.epoch_num_every_generation = 20
        self.current_population = None
        self.population_size = 100
        assert(self.epoch_num_every_generation > 2)

        # build nsga2 evolution algorithm
        mutation_control = EvolutionMutation(multi_points=-1,
                                             max_generation=self.max_generation,
                                             k0=1.0,
                                             k1=1.0,
                                             method='based_matrices',
                                             adaptive=True)
        crossover_control = EvolutionCrossover(multi_points=8,
                                               max_generation=self.max_generation,
                                               k0=1.0,
                                               k1=0.8,
                                               method='based_matrices',
                                               size=self.population_size//2)
        self.evolution_control = Nsga2(ModelProblem('MINIMIZE',
                                                    alpha=-0.07,
                                                    beta=-0.07,
                                                    threshold=self.architecture_objective_cost),
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
        # batched_sampling = None

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

        # 8.1.step compute architecture loss
        optim_cost = None
        sampled_cost = None
        pruned_cost = None
        for cost, cost_eval in self.architecture_cost_evaluators.items():
            sampled_cost, pruned_cost = cost_eval.get_costs(self.architecture(sampling, active), running_graph)

            if cost == self.architecture_cost_optimization:
                optim_cost = sampled_cost

        # evaluate every individual performance in population (only in Evaluation Stage)
        if not self.training:
            # 使用验证集进行评估个体优劣
            population_lock.acquire()
            for arch_index, sample_index in zip(batched_archs_index, list(range(batch_size))):
                evaluation_count = self.current_population.population[arch_index].evaluation_count
                pre_accuracy = self.current_population.population[arch_index].values[0]
                self.current_population.population[arch_index].values[0] = (pre_accuracy * (evaluation_count-1) + model_accuracy[sample_index].item() * self.current_population.population[arch_index].discount)/float(evaluation_count)
                self.current_population.population[arch_index].values[1] = sampled_cost[sample_index].item()
                self.evolution_control.problem.calculateObjectives(self.current_population.population[arch_index])

                print("architecture index %d evaluation count %d"%(arch_index, evaluation_count))
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

    def save_architecture(self, folder=None, name=None):
        if not os.path.exists(folder):
            os.makedirs(folder)

        # 1.step save current population
        if self.epoch % self.epoch_num_every_generation == 0 and \
                len(self.current_population.pareto_front) > 0:
            for individual in self.current_population.pareto_front:
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

                # 3.step get architecture parameter number
                parameter_num = 0
                for node in self.traversal_order:
                    sampled_state = graph.node[node]['sampled']
                    cur_node = graph.node[node]
                    parameter_num += self.blocks[cur_node['module']].get_param_num(None)[sampled_state]

                # 4.step save architecture
                architecture_info = ''
                if self.architecture_cost_optimization == 'comp':
                    architecture_info = 'flops_%d'%int(individual.values[1])
                elif self.architecture_cost_optimization == 'latency':
                    architecture_info = 'latency_%0.2f'%individual.values[1]
                else:
                    architecture_info = 'para_%d'%int(individual.values[1])

                architecture_tag = 'epoch_%d_accuray_%0.4f_%s_params_%d.architecture'%(self.epoch,
                                                                                       individual.values[0],
                                                                                       architecture_info,
                                                                                       int(parameter_num))
                architecture_path = os.path.join(folder, architecture_tag)
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

        # if self.use_preload_architecture:
        #     val = self.net.node[node_name]['sampled']
        #     node_sampling = torch.Tensor().resize_(*sampling_dim).fill_(val)
        #     node_sampling = Variable(node_sampling, requires_grad=False)
        #     return node_sampling

        node = self.net.node[node_name]
        node_sampling = batched_sampling[:, node['sampling_param']].contiguous().view(sampling_dim)

        return node_sampling

    def _sample_archs(self, batch_size, device):
        batch_size = int(batch_size)

        # 决定是否需要启动产生新种群(仅在训练阶段，进行新种群产生)
        mini_population = []
        population_lock.acquire()
        if self.epoch % self.epoch_num_every_generation == 0 and \
                self.current_population.update_population_flag and \
                self.training:
            # update population generation
            self.current_population.current_genration = self.epoch // self.epoch_num_every_generation
            self.evolution_control.mutation_controler.generation = self.current_population.current_genration
            self.evolution_control.crossover_controler.generation = self.current_population.current_genration

            print('generate pareto front')
            # 新完成训练的种群 (parent + children)
            parent_population = copy.deepcopy(self.current_population)
            # # 前代精英种群
            # parent_population.extend(copy.deepcopy(self.current_population.pareto_front))
            # 获得当代精英种群
            self.current_population.pareto_front = \
                self.evolution_control.evolve(parent_population,
                                              target_size=self.population_size).population

            print('pareto front size %d'%(len(self.current_population.pareto_front)))
            print('generate population for generation %d'%(self.current_population.current_genration))
            # 产生新种群 (crossover and mutation)
            candidate_elite_population = Population()
            candidate_elite_population.population = copy.deepcopy(self.current_population.pareto_front)

            # 交叉
            print('crossover process')
            candidate_elite_population = \
                self.evolution_control.crossover_controler.population_crossover(
                    population=candidate_elite_population,
                    graph=self.net,
                    blocks=self.blocks)

            # 变异
            print('mutation process')
            candidate_elite_population = \
                self.evolution_control.mutation_controler.population_mutate(
                    population=candidate_elite_population,
                    graph=self.net,
                    blocks=self.blocks)

            # parent.pareto_front + offsprings
            children_size = len(candidate_elite_population.population)
            parent_size = len(self.current_population.pareto_front)
            self.current_population.population = candidate_elite_population.population
            self.current_population.population.extend(copy.deepcopy(self.current_population.pareto_front))
            print('population size %d (children %d, parent %d)for generation %d'%(len(self.current_population.population),
                                                                                  children_size,
                                                                                  parent_size,
                                                                                  self.current_population.current_genration))

            # 候选精英种群初始化
            for individual_index in range(children_size+parent_size):
                self.current_population.population[individual_index].id = individual_index
                self.current_population.population[individual_index].is_selected = False
                self.current_population.population[individual_index].selected_count = 0
                self.current_population.population[individual_index].evaluation_count = 0
                self.current_population.population[individual_index].objectives[0] = 0
                self.current_population.population[individual_index].objectives[1] = 0
                self.current_population.population[individual_index].values[0] = 0
                self.current_population.population[individual_index].values[1] = 0
                if individual_index < children_size:
                    self.current_population.population[individual_index].discount = 1.0
                else:
                    self.current_population.population[individual_index].discount = 0.7
                self.current_population.population[individual_index].type = 'children'

            self.current_population.update_population_flag = False

        # next epoch, check whether need to update population
        if (self.epoch + 1) % self.epoch_num_every_generation == 0 and \
                not self.current_population.update_population_flag and \
                self.training:
            self.current_population.update_population_flag = True

        # 从种群中采样当前批次的个体
        # 1.step 从未被选中的个体中挑选
        candidate_individual_index = [individual.id for individual in self.current_population.population if not individual.is_selected]
        if len(candidate_individual_index) > batch_size:
            candidate_individual_index = np.random.choice(candidate_individual_index,
                                                          batch_size,
                                                          replace=False).flatten().tolist()

        for individual_index in candidate_individual_index:
            self.current_population.population[individual_index].is_selected = True
            if self.training:
                self.current_population.population[individual_index].selected_count += 1
            else:
                self.current_population.population[individual_index].evaluation_count += 1

            mini_population.append(self.current_population.population[individual_index])

        # 2.step 随机挑选
        if len(mini_population) < batch_size:
            remained_num = batch_size-len(mini_population)
            while remained_num > len(self.current_population.population):
                mini_population.extend(np.random.choice(self.current_population.population,
                                                        len(self.current_population.population),
                                                        replace=False).flatten().tolist())
                remained_num = remained_num - len(self.current_population.population)

            if remained_num > 0:
                mini_population.extend(np.random.choice(self.current_population.population,
                                                        remained_num,
                                                        replace=False).flatten().tolist())

            for individual in mini_population:
                if self.training:
                    individual.selected_count += 1
                else:
                    individual.evaluation_count += 1

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

    def preprocess(self):
        if self.current_population is None:
            # 随机产生初代种群
            self.current_population = Population()
            self.current_population.current_genration = 0
            for index in range(self.population_size):
                me = self.evolution_control.problem.generateIndividual()
                me.id = index
                me.features = self.randomSamplingArchitecture()
                me.values[0] = 0
                me.values[1] = 0
                me.is_selected = False
                me.selected_count = 0
                me.evaluation_count = 0
                me.discount = 1.0
                me.type = 'parent'
                self.current_population.population.append(me)

            self.current_population.update_population_flag = False

    def plot(self, path=None):
        if not os.path.exists(path):
            os.makedirs(path)

        plt.figure()
        plt.title('PARETO OPTIMAL for %d GENERATION'%(self.current_population.current_genration))
        # pareto optimal
        # axis x (flops/latency)
        if self.epoch % self.epoch_num_every_generation == 0 and \
                len(self.current_population.pareto_front) > 0:
            x = [individual.values[1] for individual in self.current_population.pareto_front]
            if self.architecture_cost_optimization == 'comp':
                x = [v / 1000000 for v in x]
            elif self.architecture_cost_optimization == 'param':
                x = [v / 1000000 for v in x]

            # axis y (objective)
            y = [individual.objectives[0] for individual in self.current_population.pareto_front]

            x_min = np.min(x)
            x_max = np.max(x)

            y_min = np.min(y)
            y_max = np.max(y)

            plt.xlim((x_min, x_max))
            plt.ylim((y_min, y_max))

            if self.architecture_cost_optimization == 'comp':
                plt.xlabel('MULADD/FLOPS (M - 10e6)')
            elif self.architecture_cost_optimization == 'latency':
                plt.xlabel('LATENCY (ms)')
            else:
                plt.xlabel("PARAMETER (M - 10e6)")

            plt.ylabel('ACCURACY')
            plt.scatter(x=x, y=y, c='r', marker='o')

            for x_p, y_p, individual in zip(x, y, self.current_population.pareto_front):
                plt.text(x_p, y_p, '%0.4f'%individual.values[0])

            if path is None:
                path = './'

            plt.savefig(os.path.join(path, 'generation_%d_pareto_optimal.png'%(self.epoch//self.epoch_num_every_generation)))
            plt.close()

    def __str__(self):
        model_descr = 'Model:{}\n\t{} nodes\n\t{} blocks\n\t{} parametrized layers\n\t{} computation steps\n\t{} parameters\n\t{} meta-params'
        return model_descr.format(type(self).__name__,
                                  self.graph.number_of_nodes(),
                                  len(self.blocks),
                                  self.n_layers,
                                  self.n_comp_steps,
                                  sum(i.numel() for i in self.parameters()),
                                  len(self.sampling_parameters))