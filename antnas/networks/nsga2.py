# -*- coding: UTF-8 -*-
# @Time    : 2019/1/16 7:11 PM
# @File    : nsga2.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import copy
from antnas.networks.bayesian import *
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import torch
import math
import random
import functools
from antnas.networks.bayesian import *
from functools import partial


class Population(object):
  """Represents population - a group of Individuals,
  can merge with another population"""

  def __init__(self):
    self.population = []
    self.fronts = []
    self.current_genration = 0
    self.update_population_flag = True
    self.pareto_front = []

  def __len__(self):
    return len(self.population)

  def __iter__(self):
    """Allows for iterating over Individuals"""

    return self.population.__iter__()

  def extend(self, new_individuals):
    """Creates new population that consists of
    old individuals ans new_individuals"""

    self.population.extend(new_individuals)


class Individual(object):
  """Represents one individual"""

  def __init__(self):
    self.rank = None
    self.crowding_distance = None
    self.dominated_solutions = set()
    self.features = None
    self.devices = None
    self.objectives = None
    self.dominates = None
    self.accuracy = 0.0

  def set_objectives(self, objectives):
    self.objectives = objectives


class Problem(object):
  def __init__(self):
    self.max_objectives = None
    self.min_objecives = None
    self.problem_type = None

  def calculateObjectives(self, individual):
    raise NotImplementedError


class Nsga2(object):
  def __init__(self,
               problem,
               mutation_op,
               crossover_op,
               num_of_generations=100,
               callback=None,
               using_bayesian=False):
    self.mutation_controler = mutation_op
    self.crossover_controler = crossover_op

    self.solution = []
    self.multi_objects = []
    self.problem = problem
    self.num_of_generations = num_of_generations
    self.callback = callback
    self.using_bayesian = using_bayesian
    self._era = 0

  @property
  def era(self):
    return self._era

  @era.setter
  def era(self, val):
    self._era = val

  def fast_nondominated_sort(self, population):
    population.fronts = []
    population.fronts.append([])

    # clear
    for individual in population:
      individual.domination_count = 0
      individual.dominated_solutions = set()
      individual.rank = 0

    # make statistic
    for individual in population:
      individual.domination_count = 0
      individual.dominated_solutions = set()

      for other_individual in population:
        if individual.dominates(other_individual):
          individual.dominated_solutions.add(other_individual)
        elif other_individual.dominates(individual):
          individual.domination_count += 1

      if individual.domination_count == 0:
        population.fronts[0].append(individual)
        individual.rank = 0

    i = 0
    while len(population.fronts[i]) > 0:
      temp = []
      for individual in population.fronts[i]:
        for other_individual in individual.dominated_solutions:
          other_individual.domination_count -= 1
          if other_individual.domination_count == 0:
            other_individual.rank = i + 1
            temp.append(other_individual)
      i = i + 1
      population.fronts.append(temp)

  def crowding_operator(self, individual, other_individual):
    if (individual.rank < other_individual.rank) or \
            ((individual.rank == other_individual.rank) and (
                    individual.crowding_distance > other_individual.crowding_distance)):
      return 1
    else:
      return -1

  def __tournament(self, population):
      participants = random.sample(population, self.tournament_size)
      best = None
      for participant in participants:
          if best is None or self.crowding_operator(participant, best) == 1:
              best = participant

      return best

  def calculate_crowding_distance(self, front):
    if len(front) > 0:
      solutions_num = len(front)
      for individual in front:
        individual.crowding_distance = 0

      for m in range(len(front[0].objectives)):
        front = sorted(front, key=lambda x: x.objectives[m])

        front[0].crowding_distance = self.problem.max_objectives[m]
        front[solutions_num - 1].crowding_distance = self.problem.max_objectives[m]

        # front[0].crowding_distance = 4444444444444
        # front[solutions_num - 1].crowding_distance = 4444444444

        for index, value in enumerate(front[1:solutions_num - 1]):
          front[index].crowding_distance += \
            (front[index + 1].objectives[m] - front[index - 1].objectives[m]) / (
                    self.problem.max_objectives[m] - self.problem.min_objectives[m] + 0.000000001)

  def __create_children(self, population, **kwargs):
    kwargs.update({'problem': self.problem})
    children_population = None
    if self.crossover_controler is not None:
      # crossover (交叉)
      print('[nsga2/create_children] crossover population')
      crossover_population = \
        self.crossover_controler.crossover(population=population, **kwargs)
      children_population = crossover_population
      
    if self.mutation_controler is not None:
      # mutation（变异）
      print('[nsga2/create_children] mutation population')
      mutation_population = \
        self.mutation_controler.mutate(population=children_population, **kwargs)
      children_population = mutation_population

    # recalculate objectives
    print('[nsga2/create_children] caculate population objectives')
    self.problem.calculateBatchObjectives(children_population.population)

    # return children population
    return children_population

  def evolve(self, population, **kwargs):
    population_size = len(population)
    # # 1.step compute nondominated_sort and crowding distance
    # self.fast_nondominated_sort(self.population)
    # for front in self.population.fronts:
    #   self.calculate_crowding_distance(front)

    # 2.step generate next children generation
    print('[nsga2/evolve] create children')
    children = self.__create_children(population, **kwargs)

    # 3.step environment pooling
    for i in range(self.num_of_generations):
      print('[nsga2/evolve] evolve generation %d' % i)

      # 3.1.step expand population
      print('[nsga2/evolve] extend population')
      population.extend(children)

      # 3.2.step re-fast-nondominated-sort
      print('[nsga2/evolve] nondominated sort')
      self.fast_nondominated_sort(population)

      new_population = Population()
      front_num = 0
      while len(new_population) + len(population.fronts[front_num]) < population_size:
        self.calculate_crowding_distance(population.fronts[front_num])
        new_population.extend(population.fronts[front_num])
        front_num += 1

      # 3.3.step sort by crowding
      self.calculate_crowding_distance(population.fronts[front_num])
      population.fronts[front_num] = sorted(population.fronts[front_num], key=functools.cmp_to_key(self.crowding_operator), reverse=True)
      new_population.extend(population.fronts[front_num][0:population_size - len(new_population)])

      population = new_population
      population.fronts = []
      for individual in population:
        individual.domination_count = 0
        individual.dominated_solutions = set()
        individual.rank = 0

      # for the last genration, dont generate children
      if i != self.num_of_generations - 1:
        # 1.step create children
        print('[nsga2/evolve] create children for generation %d'%i)
        children = self.__create_children(population, **kwargs)

        # 2.step using_bayesian to expand
        if self.using_bayesian:
          print('[nsga2/evolve] bayesian process')
          acq_num = (int)(0.2 * population_size)
          acq_num = acq_num if acq_num > 0 else population_size
          
          print('predict elite children (size=%d) by Bayesian Optimization'%acq_num)
          bo = BayesianOptimizer(0.000000001, BOAccuracy(), None, 2.576)
          x = []
          y = []
          
          # 2.1.step fill population
          for c in population:
            x.append(c.features)
            y.append(1.0-c.objectives[0])
          
          # 2.2.step fill children
          for c in children:
            x.append(c.features)
            y.append(1.0-c.objectives[0])
          
          # 2.3.step fit
          print('fit gaussian process')
          bo.fit(x, y)

          # partial(network.sample_arch, comp_min=, comp_max=)
          # 2.4.step acq some
          bayesian_population = Population()
          
          min_constraint = 0
          region_seg = 0
          network = kwargs['network']

          if network.cost_evaluation == "comp":
            region_seg = (network.arch_objective_comp_max-network.arch_objective_comp_min)/acq_num
            min_constraint = network.arch_objective_comp_min
          elif network.cost_evaluation == "latency":
            region_seg = (network.arch_objective_latency_max-network.arch_objective_latency_min)/acq_num
            min_constraint = network.arch_objective_latency_min
          elif network.cost_evaluation == "param":
            region_seg = (network.arch_objective_param_max-network.arch_objective_param_min)/acq_num
            min_constraint = network.arch_objective_param_min
          
          for acq_i in range(acq_num):
            acq_min_constraint = min_constraint + region_seg*acq_i
            acq_max_constraint = min_constraint + region_seg*(acq_i+1)
            network_arc_sampling_func = None
            if network.cost_evaluation == "comp":
              network_arc_sampling_func = \
                functools.partial(network.sample_arch,
                                  comp_min=acq_min_constraint,
                                  comp_max=acq_max_constraint)
            elif network.cost_evaluation == "latency":
              network_arc_sampling_func = \
                functools.partial(network.sample_arch,
                                  latency_min=acq_min_constraint,
                                  latency_max=acq_max_constraint)
            elif network.cost_evaluation == "param":
              network_arc_sampling_func = \
                functools.partial(network.sample_arch,
                                  param_min=acq_min_constraint,
                                  param_max=acq_max_constraint)

            print('get suggestion structure %d'%acq_i)
            suggestion_val, _ = bo.optimize_acq(network_arc_sampling_func, x, y)
            if suggestion_val is None:
              print('fail to find suggestion structure %d'%acq_i)
              continue

            print('success to find suggestion structure %d' % acq_i)
            new_individual = self.problem.generateIndividual()
            new_individual.features = suggestion_val

            print('compute suggestion structure objectives')
            # self.problem.calculateObjectives(new_individual)
            bayesian_population.population.append(new_individual)

          self.problem.calculateBatchObjectives(bayesian_population.population)
          children.extend(bayesian_population)
      
      if self.callback is not None:
        print('[nsga2/evolve] callback')
        self.callback(population, i, self.era)

    return population


# test nsga2
class ZDT1(Problem):
  def __init__(self, goal='MINIMIZE'):
    super(ZDT1, self).__init__()
    self.max_objectives = [None, None]
    self.min_objectives = [None, None]
    self.goal = goal

  def generateIndividual(self):
    individual = Individual()
    individual.features = []

    min_x = 0
    max_x = 1

    individual.features.append(min_x + (max_x - min_x) * random.random())
    individual.features.extend([min_x + (max_x - min_x) * random.random() for _ in range(10)])

    individual.objectives = [None, None]
    individual.dominates = functools.partial(self.__dominates, individual1=individual)
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
    return m.features[0]

  def __f2(self, m):
    sigma = sum(np.array(m.features)[1:])
    g = 1 + sigma * 9 / (30 - 1)
    h = 1 - (self.__f1(m) / g) ** 2
    return g * h

def callback_func(population, i):
  function1_values = [m.objectives[0] for m in population.population]
  function2_values = [m.objectives[1] for m in population.population]
  function1 = [i for i in function1_values]
  function2 = [j for j in function2_values]
  plt.xlabel('Function 1', fontsize=15)
  plt.ylabel('Function 2', fontsize=15)
  plt.scatter(function1, function2, c='r')
  # plt.savefig('aa.png')
  plt.show()


if __name__ == '__main__':
  class _ZDT1Mutation(object):
    def __init__(self):
      pass

    def mutate(self, population, **kwargs):
      min_x = 0
      max_x = 1
      for individual in population:
        individual.features = [min_x + (max_x - min_x) * random.random() for _ in range(11)]
      return population

  problem = ZDT1('MINIMIZE')
  ss = Nsga2(problem, _ZDT1Mutation(), None,callback=callback_func)

  population = Population()
  num_of_individuals = 20
  for _ in range(num_of_individuals):
    individual = problem.generateIndividual()
    problem.calculateObjectives(individual)
    population.population.append(individual)

  new_population = population
  solution = ss.evolve(new_population)

