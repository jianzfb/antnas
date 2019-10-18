# -*- coding: UTF-8 -*-
# @Time    : 2019/1/16 7:11 PM
# @File    : nsga2.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import math
import random
import functools
import copy
from nas.networks.bayesian import *
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import torch


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
    self.objectives = None
    self.values = None
    self.dominates = None
    self.id = None
    self.type = 'parent'
    self.is_selected = False
    self.selected_count = 0
    self.evaluation_count = 0

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
               tournament_size=2):
    self.mutation_controler = mutation_op
    self.crossover_controler = crossover_op

    self.solution = []
    self.multi_objects = []
    self.problem = problem
    self.tournament_size = tournament_size

  def fast_nondominated_sort(self, population):
    population.fronts = []
    population.fronts.append([])

    # clear
    for individual in population:
      individual.domination_count = 0
      individual.dominated_solutions = set()
      individual.rank = None

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

  def calculate_crowding_distance(self, front):
    if len(front) > 0:
      solutions_num = len(front)
      for individual in front:
        individual.crowding_distance = 0

      for m in range(len(front[0].objectives)):
        front = sorted(front, key=lambda x: x.objectives[m])

        # front[0].crowding_distance = self.problem.max_objectives[m]
        # front[solutions_num - 1].crowding_distance += self.problem.max_objectives[m]

        front[0].crowding_distance = 4444444444444
        front[solutions_num - 1].crowding_distance = 4444444444

        for index, value in enumerate(front[1:solutions_num - 1]):
          front[index].crowding_distance += (front[index + 1].objectives[m] - front[index - 1].objectives[m]) / (
                    self.problem.max_objectives[m] - self.problem.min_objectives[m])

  def create_children(self, population, graph, blocks, architecture_index, architecture_cost_func):
    # gp model building
    x = []
    y = []
    for p in population.population:
      x.append(p.features)
      y.append(p.values[0])

    gp = BayesianOptimizer(0.000000001, None, 0.1, 2.576)
    gp.fit(x, y)

    population_clone = copy.deepcopy(population)
    # set individual tag (children)
    for individual in population_clone.population:
      individual.type = 'children'

    if self.crossover_controler is not None:
      # 交叉
      population_clone = \
        self.crossover_controler.population_crossover(population=population_clone,
                                                      graph=graph,
                                                      blocks=blocks)
    if self.mutation_controler is not None:
      # 变异
      population_clone = \
        self.mutation_controler.population_mutate(population=population_clone,
                                                  graph=graph,
                                                  blocks=blocks)
    # using gaussian process, predict accuracy
    for p in population_clone.population:
      p.values[0] = float(gp.predict(np.array(p.features).reshape(1, -1))[0])

    # compute architecture cost
    archs = []
    traverse_nodes = list(nx.topological_sort(graph))
    for p in population_clone.population:
      archs.append([0 for _ in range(len(traverse_nodes))])
      for node_name in traverse_nodes:
        archs[-1][architecture_index[node_name]] = p.features[graph.node[node_name]['sampling_param']]

    archs = torch.as_tensor(archs).cpu()
    archs = torch.transpose(archs, 0, 1)
    archs_cost = architecture_cost_func(archs, graph)
    for p_i, p in enumerate(population_clone.population):
      p.values[1] = float(archs_cost[p_i].item())

    # calculatge objectives
    for p in population_clone.population:
      self.problem.calculateObjectives(p)

    # return children population
    return population_clone

  def evolve(self,
             population,
             target_size,
             graph=None,
             blocks=None,
             architecture_index=None,
             architecture_cost_func=None,
             children_population=None):
    # 1.step compute nondominated_sort and crowding distance
    self.fast_nondominated_sort(population)
    for front in population.fronts:
      self.calculate_crowding_distance(front)

    # environment pooling population (parent + children)
    expand_population = population

    # 2.step generate next children generation
    children = children_population
    if children is None and graph is not None:
      children = self.create_children(population, graph, blocks, architecture_index, architecture_cost_func)

    # 3.step environment pooling
    # 3.1.step expand population
    if children is not None:
      expand_population.extend(children)

    # 3.2.step re-fast-nondominated-sort
    self.fast_nondominated_sort(expand_population)

    # 3.3.step select elite into next population
    front_num = 0
    # next elite population
    new_population = Population()
    while len(new_population) + len(expand_population.fronts[front_num]) <= target_size:
      new_population.extend(expand_population.fronts[front_num])
      front_num += 1
      if front_num == len(expand_population.fronts):
        break

    if front_num == len(expand_population.fronts):
      return new_population

    if len(new_population) < target_size:
      self.calculate_crowding_distance(expand_population.fronts[front_num])
      expand_population.fronts[front_num] = sorted(expand_population.fronts[front_num],
                                                 key=functools.cmp_to_key(self.crowding_operator),
                                            reverse=True)
      new_population.extend(expand_population.fronts[front_num][0:target_size - len(new_population)])
    return new_population


# test nsga2
class ZDT1(Problem):
  def __init__(self, goal='MAXIMIZE'):
    super(ZDT1, self).__init__()
    self.max_objectives = [None, None]
    self.min_objectives = [None, None]
    self.goal = goal

  def generateIndividual(self):
    individual = Individual()
    individual.features = []

    min_x = -10
    max_x = 10

    individual.features.append(min_x + (max_x - min_x) * random.random())
    individual.features.append(None)
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
    return m.features[0]**2

  def __f2(self, m):
    value = (m.features[0]-2)**2
    return value


if __name__ == '__main__':
  class _ZDT1Mutation(object):
    def __init__(self):
      pass

    def mutate(self):
      min_x = -10
      max_x = 10
      v = min_x+(max_x-min_x)*random.random()
      return v

  problem = ZDT1('MAXIMIZE')
  ss = Nsga2(problem, _ZDT1Mutation(), None)


  population = Population()
  num_of_individuals = 20
  for _ in range(num_of_individuals):
    individual = problem.generateIndividual()
    problem.calculateObjectives(individual)
    population.population.append(individual)

  new_population = population
  for index in range(46):
    if index > 0:
      children_population = Population()
      for _ in range(num_of_individuals):
        individual = problem.generateIndividual()
        problem.calculateObjectives(individual)
        children_population.population.append(individual)

      new_population = ss.evolve(new_population, children_population=None)

    function1_values = [m.objectives[0] for m in new_population.population]
    function2_values = [m.objectives[1] for m in new_population.population]
    function1 = [i  for i in function1_values]
    function2 = [j  for j in function2_values]
    plt.xlabel('Function 1', fontsize=15)
    plt.ylabel('Function 2', fontsize=15)
    plt.scatter(function1, function2, c='r')
    # plt.savefig('aa.png')
    plt.show()