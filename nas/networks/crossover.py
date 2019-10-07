# -*- coding: UTF-8 -*-
# @Time    : 2019/1/26 12:48 PM
# @File    : crossover.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
import random
import networkx as nx
from nas.networks.nsga2 import *
import copy


class CrossOver(object):
  def __init__(self, crossover_type, multi_points, adaptive=True, **kwargs):
    self.crossover_type = crossover_type
    self.multi_points = multi_points
    self.adaptive = adaptive

    self._generation = 0
    self.max_generation = kwargs.get('max_generation', 1)
    self.k0 = kwargs.get('k0', 0.2)
    self.k1 = kwargs.get('k1', 1.0)
    self.size = kwargs.get('size', 0)

  @property
  def generation(self):
    return self._generation

  @generation.setter
  def generation(self, val):
    self._generation = val

  def _crossover_based_matrices(self, *args, **kwargs):
      # fitness_values: [(index, fitness, gene, rate), (index, fitness, gene, rate), ...]
      fitness_values = kwargs['fitness_values']
      N = len(fitness_values)
      M = len(fitness_values[0][2])

      C = np.zeros((N, 1))  # fitness cumulative probability of chromosome i,
      # can be considered as an information measure of chromosome i
      ordered_fitness = [(f[0], f[1]) for f in fitness_values]
      ordered_fitness = sorted(ordered_fitness, key=lambda x: x[1])
      ordered_fitness_values = np.array([m[1] for m in ordered_fitness])
      probability_fitness = ordered_fitness_values / np.sum(ordered_fitness_values)

      c_sum = 0.0
      for a, b in zip(ordered_fitness, probability_fitness):
          c_sum += b
          C[a[0], 0] = c_sum

      A = np.zeros((N, M))
      for n in range(N):
          A[n, :] = np.array(fitness_values[n][2])

      # which position in chromosome i contribute fitness
      sigma = np.sum(np.power(A - np.mean(A, 0), 2.0) * C, 0) / np.sum(C)
      # remove those fixed position
      position_contribution = np.where(sigma > 0.00001)
      if position_contribution[0].size == 0:
          print('couldnt finding crossover locs because of sigma')
          return []

      probability_contribution = 1.0 - sigma[position_contribution] / (np.sum(sigma[position_contribution]) + 0.000000001)
      probability_contribution = probability_contribution / np.sum(probability_contribution)

      # transform to probability
      chromosome_probability = C/np.sum(C)
      chromosome_probability = chromosome_probability.flatten()

      # 计算chromosome之间的距离
      H = np.zeros((N, N))
      for ii in range(N):
          for jj in range(N):
              if ii == jj:
                H[ii, jj] = 0.0
              else:
                H[ii, jj] = (np.array(fitness_values[ii][2], dtype=np.int) != np.array(fitness_values[jj][2], dtype=np.int)).sum()

      crossover_result = []
      crossover_num = self.size if self.size > 0 else N//2
      for _ in range(crossover_num):
        # selecting first chromosome
        first_chromosome_index = np.random.choice(list(range(N)), p=chromosome_probability)

        # selecting second chromosome
        PCII = H[first_chromosome_index, :]/np.sum(H[first_chromosome_index, :])
        second_chromosome_index = np.random.choice(list(range(N)), p=PCII)

        # selecting
        crossover_pos = np.random.choice(position_contribution[0].flatten().tolist(),
                                         size=self.multi_points,
                                         p=probability_contribution,
                                         replace=False)
        if len(crossover_pos) > 0:
            crossover_result.append((first_chromosome_index, second_chromosome_index, crossover_pos.tolist()))

      if len(crossover_result) == 0:
          print('couldnt finding crossover locs because of others')

      return crossover_result

  def _crossover_simple(self, *args, **kwargs):
    # fitness_values: [(index, fitness, gene, rate), (index, fitness, gene, rate), ...]
    fitness_values = kwargs['fitness_values']
    N = len(fitness_values)
    individual_indexes = [m[0] for m in fitness_values]

    crossover_result = []
    feature_length = len(fitness_values[0][2])
    for _ in range(N):
      # 1.step select two randomly
      first, second = np.random.choice(individual_indexes, size=2, replace=False)

      # 2.step select crossover region
      crossover_pos = np.random.choice(list(range(feature_length)), size=self.multi_points, replace=False)
      crossover_result.append((first, second, crossover_pos.tolist()))

    return crossover_result

  def adaptive_crossover(self, *args, **kwargs):
    if self.crossover_type.lower() == 'simple':
      return self._crossover_simple(*args, **kwargs)
    elif self.crossover_type.lower() == 'based_matrices':
      return self._crossover_based_matrices(*args, **kwargs)

    return None


class EvolutionCrossover(CrossOver):
  def __init__(self,
               multi_points,
               max_generation,
               k0,
               k1,
               method='simple',
               size=0):
    super(EvolutionCrossover, self).__init__(method,
                                             multi_points,
                                             adaptive=True,
                                             max_generation=max_generation,
                                             k0=k0,
                                             k1=k1,
                                             size=size)

  def population_crossover(self, *args, **kwargs):
    population = kwargs['population']

    fitness_values = []
    for individual_index, individual in enumerate(population.population):
      fitness_values.append((individual_index,            # index
                             individual.objectives[0],    # accuracy
                             individual.features,         # feature
                             None))

    # finding crossover region
    crossover_individuals = self.adaptive_crossover(fitness_values=fitness_values)

    # cross gene infomation
    crossover_population = Population()
    for crossover_suggestion in crossover_individuals:
        first_individual_index, second_individual_index, crossover_region = crossover_suggestion
        individual_clone = copy.deepcopy(population.population[first_individual_index])

        for loc in crossover_region:
            individual_clone.features[loc] = population.population[second_individual_index].features[loc]
        crossover_population.population.append(individual_clone)

    return crossover_population
