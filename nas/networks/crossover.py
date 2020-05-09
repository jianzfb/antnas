# -*- coding: UTF-8 -*-
# @Time    : 2019/1/26 12:48 PM
# @File    : crossover.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from nas.networks.nsga2 import *
from nas.component.NetworkBlock import *
import copy
import numpy as np
import random
import networkx as nx


class CrossOver(object):
    expore_gene_pos = []

    def __init__(self, crossover_type, multi_points, adaptive=True, **kwargs):
        self.crossover_type = crossover_type
        self.multi_points = multi_points
        self.adaptive = adaptive
        
        self._generation = 0
        self.max_generation = kwargs.get('max_generation', 1)
        self.k0 = kwargs.get('k0', 0.2)
        self.k1 = kwargs.get('k1', 1.0)
        self.size = kwargs.get('size', 0)
        self.hierarchical = kwargs.get('hierarchical', [])

    @property
    def generation(self):
        return self._generation

    @generation.setter
    def generation(self, val):
        self._generation = val

    def _crossover_based_matrices(self, *args, **kwargs):
        # fitness_values: [(index, fitness, gene, rate), (index, fitness, gene, rate), ...]
        fitness_values = kwargs['fitness_values']

        N = len(fitness_values)           # 个数
        M = len(fitness_values[0][2])     # 基因长度

        C = np.zeros((N, 1))  # fitness cumulative probability of chromosome i,
        # # can be considered as an information measure of chromosome i
        # ordered_fitness = [(f[0], f[1]) for f in fitness_values]
        # ordered_fitness = sorted(ordered_fitness, key=lambda x: x[1])
        # ordered_fitness_values = np.array([m[1] for m in ordered_fitness])
        # probability_fitness = ordered_fitness_values / np.sum(ordered_fitness_values)
        #
        # c_sum = 0.0
        # for a, b in zip(ordered_fitness, probability_fitness):
        #     c_sum += b
        #     C[a[0], 0] = c_sum
        #
        for n in range(N):
            C[n, 0] = fitness_values[n][1]

        # transform to probability
        C_SUM = np.sum(C)
        if C_SUM < 0.0000001:
            C = C + 0.0000001
            C_SUM = np.sum(C)

        chromosome_probability = C / C_SUM
        chromosome_probability = chromosome_probability.flatten()

        A = np.zeros((N, M))
        for n in range(N):
            A[n, :] = np.array(fitness_values[n][2])
        AA = A.astype(np.int32)

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
        for crossover_count in range(crossover_num):
            # selecting first chromosome
            first_chromosome_index = np.random.choice(list(range(N)), p=chromosome_probability)
    
            # selecting second chromosome
            PCII = H[first_chromosome_index, :]/np.sum(H[first_chromosome_index, :])
            second_chromosome_index = np.random.choice(list(range(N)), p=PCII)
            
            # crossover pos number
            multi_points = self.multi_points
            
            # hierarchical selection
            # stage/block/cell
            stage_i = -1
            block_num = 0
            if len(self.hierarchical) > 0:
                stage_num = len(self.hierarchical)
                stage_i = np.random.randint(0, stage_num)
                block_num = len(self.hierarchical[stage_i])

            explore_position = []
            if stage_i >= 0 and block_num > 0:
                if np.random.random() < 0.3:
                    # TODO 自适应阈值修改
                    # whole block changing
                    block_i = np.random.randint(0, block_num)
                    explore_position = self.hierarchical[stage_i][block_i]
                    multi_points = len(explore_position)
                else:
                    # random cell crossover
                    explore_position = kwargs['explore_position']

            # selecting diff gene position
            first_gene = AA[first_chromosome_index, :]
            second_gene = AA[second_chromosome_index, :]
            
            diff_gene = first_gene-second_gene
            diff_pos = np.where(diff_gene != 0)[0]
            diff_pos = list(set(diff_pos.tolist()) & set(explore_position))
            
            if multi_points > len(diff_pos):
                # try half
                multi_points = multi_points//2
            
            if multi_points > len(diff_pos):
                # try half
                multi_points = multi_points//2
            
            if multi_points > len(diff_pos) or multi_points == 0 or len(diff_pos) == 0:
                print('dont have enough diff gene position')
                continue
            
            crossover_pos = np.random.choice(diff_pos, size=multi_points, replace=False)
            print("crossover %d count first %d second %d selection %s"%(crossover_count,
                                                                        first_chromosome_index,
                                                                        second_chromosome_index,
                                                                        str(crossover_pos.tolist())))
            
            crossover_result.append((first_chromosome_index, second_chromosome_index, crossover_pos.tolist()))
    
        return crossover_result

    def _crossover(self, *args, **kwargs):
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
            return self._crossover(*args, **kwargs)
        elif self.crossover_type.lower() == 'based_matrices':
            return self._crossover_based_matrices(*args, **kwargs)


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

  def crossover(self, *args, **kwargs):
    population = kwargs['population']
    problem = kwargs['problem']
    explore_position = kwargs['explore_position']
    self.hierarchical = kwargs['hierarchical']

    fitness_values = []
    for individual_index, individual in enumerate(population.population):
      fitness_values.append((individual_index,                  # index
                             1.0-individual.objectives[0],      # accuracy
                             individual.features,               # feature
                             None))

    # finding crossover region
    crossover_individuals = \
        self.adaptive_crossover(fitness_values=fitness_values,
                                explore_position=explore_position)

    # cross gene infomation
    print('reorganize crossover population')
    crossover_population = Population()
    for crossover_suggestion in crossover_individuals:
        first_individual_index, second_individual_index, crossover_region = crossover_suggestion
        if second_individual_index is not None:
            crossover_1_individual = problem.generateIndividual()
            crossover_1_individual.features = copy.deepcopy(population.population[first_individual_index].features)
            crossover_1_individual.objectives = copy.deepcopy(population.population[first_individual_index].objectives)

            crossover_2_individual = problem.generateIndividual()
            crossover_2_individual.features = copy.deepcopy(population.population[second_individual_index].features)
            crossover_2_individual.objectives = copy.deepcopy(population.population[second_individual_index].objectives)

            for loc in crossover_region:
                crossover_1_individual.features[loc] = population.population[second_individual_index].features[loc]
                crossover_2_individual.features[loc] = population.population[first_individual_index].features[loc]

            crossover_population.population.append(crossover_1_individual)
            crossover_population.population.append(crossover_2_individual)

    if len(crossover_population) < len(population):
        select_index_list = \
            np.random.choice(list(range(len(population))),
                             size=len(population) - len(crossover_population),
                             replace=False)
        for ii in select_index_list.tolist():
            new_individual = problem.generateIndividual()
            new_individual.features = copy.deepcopy(population.population[ii].features)
            new_individual.objectives = copy.deepcopy(population.population[ii].objectives)
            crossover_population.population.append(new_individual)

    # may be larger than > original population size
    return crossover_population
