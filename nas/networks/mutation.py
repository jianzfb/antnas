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
from nas.interfaces.NetworkBlock import *


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
    N = len(fitness_values)
    M = len(fitness_values[0][2])

    C = np.zeros((N, 1))      # fitness cumulative probability of chromosome i,
                              # can be considered as an information measure of chromosome i
    ordered_fitness = [(f[0], f[1]) for f in fitness_values]
    ordered_fitness = sorted(ordered_fitness, key=lambda x: x[1])
    ordered_fitness_values = np.array([m[1] for m in ordered_fitness])
    probability_fitness = ordered_fitness_values / np.sum(ordered_fitness_values)

    gamma = 1
    if self.adaptive:
      gamma = np.exp(float(self.generation)/float(self.max_generation * self.k0) - self.k1)

    probability_fitness = np.power(probability_fitness, gamma)
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

    # which position in chromosome i should mutation
    sigma = np.sum(np.power(A - np.mean(A, 0), 2.0) * C, 0) / np.sum(C)
    position_sigma = np.where(sigma > 0.00001)
    if position_sigma[0].size == 0:
        print('couldnt finding mutation locs because of sigma')
        return []

    probability_sigma = sigma[position_sigma]
    probability_sigma = np.power(probability_sigma, gamma)
    probability_sigma = probability_sigma / (np.sum(probability_sigma) + 0.000000001)

    mutation_result = []
    for f in fitness_values:
        # mutation points number
        multi_points = self.multi_points if self.multi_points > 0 else int(alpha[f[0]] * len(position_sigma[0].flatten().tolist()))
        if multi_points > 0:
            mutation_position = np.random.choice(position_sigma[0].flatten().tolist(),
                                                 multi_points,
                                                 replace=False,
                                                 p=probability_sigma)
            mutation_result.append((f + (mutation_position.flatten().tolist(),)))
        else:
            mutation_result.append((f + (None,)))

    return mutation_result

  def _mutate_simple(self, *args, **kwargs):
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
      return self._mutate_simple(*args, **kwargs)
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

  def population_mutate(self, *args, **kwargs):
    population = kwargs['population']
    graph = kwargs['graph']
    blocks = kwargs['blocks']

    traversal_order = list(nx.topological_sort(graph))
    pos_map = {}
    for node_name in traversal_order:
        cur_node = graph.node[node_name]
        pos_map[cur_node['sampling_param']] = node_name

    fitness_values = []
    for individual_index, individual in enumerate(population.population):
      fitness_values.append((individual_index,
                             individual.objectives[0],
                             individual.features,
                             None))

    mutation_individuals = self.adaptive_mutate(fitness_values=fitness_values)

    for individual in mutation_individuals:
      if individual[-1] is not None:
        individual_index = individual[0]
        mutation_position = individual[-1]
        mutation_position = sorted(mutation_position)

        for pos in mutation_position:
            node_name = pos_map[pos]
            node = graph.node[node_name]

            if node_name.startswith("CELL") or node_name.startswith('T'):
                mutated_val = 0
                if blocks[node['module']].switch:
                    before_val = population.population[individual_index].features[pos]
                    if before_val == 0:
                        mutated_val = 1
                    else:
                        mutated_val = 0

                    population.population[individual_index].features[pos] = mutated_val
                else:
                    population.population[individual_index].features[pos] = int(np.random.randint(0, NetworkBlock.state_num, 1))

    return population
