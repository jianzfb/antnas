# -*- coding: UTF-8 -*-
# @Time    : 2019/1/26 12:47 PM
# @File    : mutation.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
import random
import networkx as nx
from nas.component.NetworkBlock import *


class Mutation(object):
  def __init__(self, mutation_type, multi_points, adaptive=True, **kwargs):
    self.adaptive = adaptive
    self.mutation_type = mutation_type
    self.multi_points = multi_points    # -1: auto (simulated annealing)
    self.max_generation = kwargs.get('max_generation', 1)
    self._generation = 0
    self.k0 = kwargs.get('k0', 0.1)
    self.k1 = kwargs.get('k1', 1.0)

  @property
  def generation(self):
    return self._generation

  @generation.setter
  def generation(self, val):
    self._generation = val

  def _mutate_based_matrices(self, *args, **kwargs):
    # fitness_values: [(index, fitness, gene, rate), (index, fitness, gene, rate), ...]
    fitness_values = kwargs['fitness_values']
    explore_position = kwargs['explore_position']

    N = len(fitness_values)
    M = len(fitness_values[0][2])

    C = np.zeros((N, 1))      # fitness cumulative probability of chromosome i,
                              # can be considered as an information measure of chromosome i
    ordered_fitness = [(f[0], f[1]) for f in fitness_values]
    ordered_fitness = sorted(ordered_fitness, key=lambda x: x[1])
    probability_fitness = np.array([m[1] for m in ordered_fitness])

    # gamma = 1
    # if self.adaptive:
    #   gamma = np.exp(float(self.generation)/float(self.max_generation * self.k0) - self.k1)
    # probability_fitness = np.power(probability_fitness, gamma)
    probability_fitness = probability_fitness / np.sum(probability_fitness)

    c_sum = 0.0
    for a, b in zip(ordered_fitness, probability_fitness):
      c_sum += b
      C[a[0], 0] = c_sum

    # which individual should mutation
    alpha = 1.0 - C     # the probability to choose which individual for mutation

    A = np.zeros((N, M))
    for n in range(N):
      A[n, :] = np.array(fitness_values[n][2])

    # 分析p(fitness|pos,state)
    AA = A.astype(np.int32)
    AA_FITNESS = np.array([f[1] for f in fitness_values])
    probability_contribution_pos = np.zeros((M))
    for m in range(M):
        m_list = AA[:, m].tolist()
        lookup = {}
        for state_i in range(NetworkBlock.state_num):
            count = m_list.count(state_i)
            lookup[state_i] = (float)(count) / (float)(N)

        for n in range(N):
            probability_contribution_pos[m] += lookup[int(AA[n, m])] * AA_FITNESS[n]

    position_contribution = explore_position
    # 获得每一基因位的变异概率
    # 基因位置对适应度的贡献（除去那些不动基因）
    probability_contribution_pos = probability_contribution_pos[position_contribution]
    probability_contribution_pos = probability_contribution_pos / np.sum(probability_contribution_pos)
    # 贡献的反
    probability_contribution_pos = 1.0 - probability_contribution_pos
    # 重新归一化
    probability_contribution_pos = probability_contribution_pos / np.sum(probability_contribution_pos)

    print("gene mutation probability at position")
    print(probability_contribution_pos)
    print(position_contribution)

    #######################################################################
    mutation_result = []
    for f_index, f in enumerate(fitness_values):
        # mutation points number
        multi_points = self.multi_points if self.multi_points > 0 else int(alpha[f[0]] * len(position_contribution))
        if multi_points > 0:
            print("multi_points %d"%(multi_points))
            mutation_position = np.random.choice(position_contribution,
                                                 multi_points,
                                                 replace=False,
                                                 p=probability_contribution_pos)
            mutation_position = mutation_position.flatten().tolist()
            mutation_state = []
            for _ in mutation_position:
              s = np.random.choice(list(range(NetworkBlock.state_num)),
                                   1,
                                   replace=False)
              mutation_state.append(int(s))

            print("individual %d mutation at %s to state %s"%(f_index, str(mutation_position), str(mutation_state)))
            mutation_result.append((f + (mutation_position, mutation_state)))
        else:
            print("individual %d mutation at none"%f_index)
            mutation_result.append((f + (None, None)))

    return mutation_result

  def _mutate(self, *args, **kwargs):
    # fitness_values: [(index, fitness, gene, rate), (index, fitness, gene, rate), ...]
    fitness_values = kwargs['fitness_values']
    gene_length = len(fitness_values[0][2])
    gene_num = len(fitness_values)

    decreasing_fitness = [(f[0], f[1]) for f in fitness_values]
    decreasing_fitness = sorted(decreasing_fitness, key=lambda x: x[1], reverse=True)

    # continue N1 mutation
    N1 = int(self.k0 * gene_num)
    # worse N3 random sampling
    N2 = gene_num - N1

    mutate_result = []
    # 1.step best N1 mutation
    for chromo_index in range(0, N1):
        mutation_locs = []
        for loc in range(gene_length):
            if random.random() < self.k1:
                mutation_locs.append(loc)

        mutate_result.append((fitness_values[decreasing_fitness[chromo_index][0]] + (mutation_locs,)))

    # 2.step worse N3 random sampling completely
    for chromo_index in range(N1, gene_num):
        mutate_result.append((fitness_values[decreasing_fitness[chromo_index][0]] + (list(range(gene_num)),)))

    return mutate_result

  def adaptive_mutate(self, *args, **kwargs):
    if self.mutation_type.lower() == 'simple':
      return self._mutate(*args, **kwargs)
    elif self.mutation_type.lower() == 'based_matrices':
      return self._mutate_based_matrices(*args, **kwargs)

    return None


class EvolutionMutation(Mutation):
  def __init__(self,
               multi_points,
               max_generation,
               k0,
               k1,
               method='simple',
               adaptive=True):
    super(EvolutionMutation, self).__init__(method,
                                            multi_points,
                                            adaptive=adaptive,
                                            max_generation=max_generation,
                                            k0=k0,
                                            k1=k1)

  def mutate(self, *args, **kwargs):
    population = kwargs['population']
    graph = kwargs['graph']
    blocks = kwargs['blocks']
    explore_position = kwargs['explore_position']

    traversal_order = list(nx.topological_sort(graph))
    pos_map = {}
    for node_name in traversal_order:
        cur_node = graph.node[node_name]
        pos_map[cur_node['sampling_param']] = node_name

    fitness_values = []
    for individual_index, individual in enumerate(population.population):
      fitness_values.append((individual_index,                      # index
                             1.0-individual.objectives[0],          # accuracy
                             individual.features,                   # feature
                             None))

    mutation_individuals = \
        self.adaptive_mutate(fitness_values=fitness_values,
                             explore_position=explore_position)

    for _, individual in enumerate(mutation_individuals):
      if individual[-1] is not None:
        individual_index = individual[0]
        mutation_position = individual[-2]
        mutation_state = individual[-1]

        for mutation_index, mutation_pos in enumerate(mutation_position):
            node_name = pos_map[mutation_pos]
            node = graph.node[node_name]

            if node_name.startswith("CELL") or node_name.startswith('T'):
                if not blocks[node['module']].structure_fixed:
                    mutated_state = mutation_state[mutation_index]
                    if mutated_state != 0 and mutated_state != 1:
                      mutated_state = 1
                    population.population[individual_index].features[mutation_pos] = int(mutated_state)
                else:
                    # cur_state = population.population[individual_index].features[pos]
                    # mutated_state = np.random.choice([a for a in list(range(NetworkBlock.state_num)) if a != cur_state], 1)
                    mutated_state = mutation_state[mutation_index]
                    population.population[individual_index].features[mutation_pos] = int(mutated_state)
                    print("individual %d mutation to %d at pos %d" % (individual_index,population.population[individual_index].features[mutation_pos], mutation_pos))
            else:
                print("shouldnt mutation at this pos")

    return population
