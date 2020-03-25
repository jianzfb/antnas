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
from nas.component.NetworkBlock import *
import copy
# from fim import eclat


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

  @property
  def generation(self):
    return self._generation

  @generation.setter
  def generation(self, val):
    self._generation = val

  def __picking_from(self, first, second, explore_transactions, frequent_item_set, explore_transactions_fis_score):
    pick_list = []

    try_times = 100
    for try_i in range(try_times):
      # random_index = np.random.randint(0, len(frequent_item_set))
      random_index = np.random.choice(list(range(len(explore_transactions_fis_score))), p=explore_transactions_fis_score)

      items, _ = frequent_item_set[random_index]
      if set(items).issubset(set(explore_transactions[first])) and (not set(items).issubset(set(explore_transactions[second]))):
        pick_list.append(random_index)

      if len(pick_list) >= 3:
        return pick_list

    for _ in range(3 - len(pick_list)):
      random_index = np.random.randint(0, len(frequent_item_set))
      pick_list.append(random_index)

    return pick_list

  def _crossover_based_pattern(self, *args, **kwargs):
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

      explore_position_index = self.expore_gene_pos
      explore_transactions = []
      explore_transactions_score = []
      for n in range(N):
        transaction = []
        for mm_index, mm in enumerate(explore_position_index):
            transaction.append(int(mm_index * NetworkBlock.state_num + int(A[n, mm])))

        explore_transactions.append(transaction)
        explore_transactions_score.append(fitness_values[n][1])

      support = 8
      frequent_item_set = eclat(explore_transactions, supp=-support, zmin=2)
      print("finding frequent item set")
      print(frequent_item_set)

      # 转换到标准 基因位-状态 对
      converted_frequent_item_set = []
      for fis in frequent_item_set:
        items, _ = fis
        excellent_gene_ps = []
        for iii in items:
          gene_p = explore_position_index[iii // NetworkBlock.state_num]
          gene_s = iii % NetworkBlock.state_num
          excellent_gene_ps.append((int(gene_p), int(gene_s)))

        converted_frequent_item_set.append(excellent_gene_ps)

      # 统计每个set的排名，根据排名建立概率分布
      explore_transactions_fis_score = []
      for fis in frequent_item_set:
        items, _ = fis
        score = 0.0
        score_count = 0
        for n in range(N):
          if set(items).issubset(set(explore_transactions[n])):
            score += explore_transactions_score[n]
            score_count += 1
        score = score / score_count
        explore_transactions_fis_score.append(score)

      explore_transactions_fis_score = np.array(explore_transactions_fis_score) / np.sum(explore_transactions_fis_score)
      print('explore frequent item set score')
      print(explore_transactions_fis_score)

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

      # 随机挑选可以交叉变异对，将没有参与交叉变异的个体，依然保留下来
      crossover_chromosome_list = []
      for crossover_count in range(crossover_num):
          # selecting first chromosome
          first_chromosome_index = np.random.choice(list(range(N)), p=chromosome_probability)

          # selecting second chromosome
          PCII = H[first_chromosome_index, :]/np.sum(H[first_chromosome_index, :])
          second_chromosome_index = np.random.choice(list(range(N)), p=PCII)

          # 挑选交叉基因位
          fis_index_list = self.__picking_from(first_chromosome_index, second_chromosome_index, explore_transactions, frequent_item_set, explore_transactions_fis_score)
          crossover_info = []
          for fis_index in fis_index_list:
            crossover_info.extend(converted_frequent_item_set[fis_index])

          print("crossover info")
          print(crossover_info)
          crossover_pos = [x[0] for x in crossover_info]

          print("crossover %d count first %d second %d selection %s"%(crossover_count,
                                                                    first_chromosome_index,
                                                                    second_chromosome_index,
                                                                    str(crossover_pos)))

          if len(crossover_pos) > 0:
              crossover_result.append((first_chromosome_index, second_chromosome_index, crossover_pos))

          crossover_chromosome_list.append(first_chromosome_index)
          crossover_chromosome_list.append(second_chromosome_index)

      for i in range(N):
          if i not in crossover_chromosome_list:
              print("no crossover %d"%i)
              crossover_result.append((i, None, None))

      if len(crossover_result) == 0:
          print('couldnt finding crossover locs because of others')

      return crossover_result

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

      # # which position in chromosome i contribute fitness
      # sigma = np.sum(np.power(A - np.mean(A, 0), 2.0) * C, 0) / np.sum(C)
      # # remove those fixed position
      # position_contribution = np.where(sigma > 0.00001)
      # if position_contribution[0].size == 0:
      #     print('couldnt finding crossover locs because of sigma')
      #     return []

      # probability_contribution = 1.0 - sigma[position_contribution] / (np.sum(sigma[position_contribution]) + 0.000000001)
      # probability_contribution = probability_contribution / np.sum(probability_contribution)

      ######################################################
      # 分析每一位置对结果的贡献
      AA = A.astype(np.int32)
      AA_FITNESS = np.array([f[1] for f in fitness_values])
      probability_contribution = np.zeros((M))
      for m in range(M):
        m_list = AA[:, m].tolist()
        lookup = {}
        for state_i in range(NetworkBlock.state_num):
            count = m_list.count(state_i)
            lookup[state_i] = (float)(count)/(float)(N)

        m_contribution = 0.0
        for n in range(N):
            m_contribution += lookup[int(AA[n, m])] * AA_FITNESS[n]

        probability_contribution[m] = m_contribution

      sigma = np.std(A, 0)
      position_contribution = np.where(sigma > 0.00001)
      if position_contribution[0].size == 0:
          print('couldnt finding crossover locs because of sigma')
          return []

      probability_contribution = probability_contribution[position_contribution]
      probability_contribution = probability_contribution / np.sum(probability_contribution)
      #######################################################

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

      # 随机挑选可以交叉变异对，将没有参与交叉变异的个体，依然保留下来
      crossover_chromosome_list = []
      for crossover_count in range(crossover_num):
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

          print("crossover %d count first %d second %d selection %s"%(crossover_count,
                                                                    first_chromosome_index,
                                                                    second_chromosome_index,
                                                                    str(crossover_pos.tolist())))

          if len(crossover_pos) > 0:
              crossover_result.append((first_chromosome_index, second_chromosome_index, crossover_pos.tolist()))

          crossover_chromosome_list.append(first_chromosome_index)
          crossover_chromosome_list.append(second_chromosome_index)

      for i in range(N):
          if i not in crossover_chromosome_list:
              print("no crossover %d"%i)
              crossover_result.append((i, None, None))

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

    # return self._crossover_based_pattern(*args, **kwargs)
    # return None


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

    fitness_values = []
    for individual_index, individual in enumerate(population.population):
      fitness_values.append((individual_index,                  # index
                             1.0-individual.objectives[0],      # accuracy
                             individual.features,               # feature
                             None))

    # finding crossover region
    crossover_individuals = self.adaptive_crossover(fitness_values=fitness_values)

    # cross gene infomation
    print('reorganize crossover population')
    crossover_population = Population()
    for crossover_suggestion in crossover_individuals:
        first_individual_index, second_individual_index, crossover_region = crossover_suggestion

        if second_individual_index is not None:
            first_individual_clone = copy.deepcopy(population.population[first_individual_index])
            second_individual_clone = copy.deepcopy(population.population[second_individual_index])

            for loc in crossover_region:
                first_individual_clone.features[loc] = population.population[second_individual_index].features[loc]
                second_individual_clone.features[loc] = population.population[first_individual_index].features[loc]

            crossover_population.population.append(first_individual_clone)
            crossover_population.population.append(second_individual_clone)
        else:
            first_individual_clone = copy.deepcopy(population.population[first_individual_index])
            crossover_population.population.append(first_individual_clone)

    # may be larger than > original population size
    return crossover_population
