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
from nas.networks.nsga2 import *
from nas.component.NetworkBlock import *


class Mutation(object):
    def __init__(self, mutation_type, multi_points, adaptive=True, **kwargs):
        self.adaptive = adaptive
        self.mutation_type = mutation_type
        self.multi_points = multi_points    # -1: auto
        self.max_generation = kwargs.get('max_generation', 1)
        self._generation = 0
        self.k0 = kwargs.get('k0', 0.1)
        self.k1 = kwargs.get('k1', 1.0)
        self.hierarchical = kwargs.get('hierarchical', [])
        self.network = kwargs.get('network', None)
        
        self.pos_map = {}
        traversal_order = list(nx.topological_sort(self.network.net))
        for node_name in traversal_order:
            cur_node = self.network.net.node[node_name]
            self.pos_map[cur_node['sampling_param']] = node_name

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

        N = len(fitness_values)             # 个数
        M = len(fitness_values[0][2])       # 基因长度

        C = np.zeros((N, 1))        # fitness cumulative probability of chromosome i,
                                    # can be considered as an information measure of chromosome i
        ordered_fitness = [(f[0], f[1]) for f in fitness_values]
        ordered_fitness = sorted(ordered_fitness, key=lambda x: x[1])
        probability_fitness = np.array([m[1] for m in ordered_fitness])
        PF_SUM = np.sum(probability_fitness)
        if PF_SUM < 0.0000001:
            probability_fitness = probability_fitness + 0.0000001
            PF_SUM = np.sum(probability_fitness)
        probability_fitness = probability_fitness / PF_SUM
        
        c_sum = 0.0
        for a, b in zip(ordered_fitness, probability_fitness):
            c_sum += b
            C[a[0], 0] = c_sum
        
        # which individual should mutation
        alpha = 1.0 - C     # the probability to choose which individual for mutation

        mutation_result = []
        for f_index, f in enumerate(fitness_values):
            # 变异比例
            mutation_ratio = alpha[f[0]]
            
            # 1.step hierarchical selection
            explore_position = kwargs['explore_position']
            # stage/block/cell
            stage_i = -1
            block_num = 0
            if len(self.hierarchical) > 0:
                stage_num = len(self.hierarchical)
                stage_i = np.random.randint(0, stage_num)
                block_num = len(self.hierarchical[stage_i])
            
            if stage_i >= 0 and block_num > 0:
                if np.random.random() < mutation_ratio:
                    # whole block changing
                    block_i = np.random.randint(0, block_num)
                    explore_position = self.hierarchical[stage_i][block_i]
                    mutation_ratio = 1.0
                    print('whole block %d mutation in hierarchical'%block_i)
                else:
                    # cell selecting in block
                    if np.random.random() < 0.7:
                        block_i = np.random.randint(0, block_num)
                        explore_position = self.hierarchical[stage_i][block_i]
                        print('local cell mutation in block %d in hierarchical'%block_i)
                    else:
                        explore_position = kwargs['explore_position']
                        print('random cell mutation')

            # 2.step mutation pos selection
            # mutation points number
            multi_points = self.multi_points if self.multi_points > 0 else int(mutation_ratio * len(explore_position))
            multi_points = min(multi_points, len(explore_position))
            if multi_points > 0:
                is_ok = False
                
                while not is_ok:
                    mutation_position = np.random.choice(explore_position, multi_points, replace=False)
                    mutation_position = mutation_position.flatten().tolist()
                    mutation_state = []
                    feature = copy.deepcopy(f[2])
                    
                    for mutation_p in mutation_position:
                        candidate_state = set(list(range(NetworkBlock.state_num)))
                        candidate_state.discard(f[2][mutation_p])
                        candidate_state = list(candidate_state)
                        
                        s = np.random.choice(candidate_state,
                                             1,
                                             replace=False)

                        node_name = self.pos_map[mutation_p]
                        node = self.network.net.node[node_name]

                        if node_name.startswith("CELL") or node_name.startswith('T'):
                            if not self.network.blocks[node['module']].structure_fixed:
                                if s != 0 and s != 1:
                                    s = 1
                        else:
                            print("shouldnt mutation at this pos %d(%s)"%(mutation_p, node_name))

                        mutation_state.append(int(s))
                        feature[mutation_p] = int(s)
                    
                    # 检查变异后基因是否满足约束条件
                    if not self.network.is_satisfied_constraint(feature):
                        continue
                    
                    is_ok = True
                    
                    print("individual %d mutation at %s to state %s"%(f_index, str(mutation_position), str(mutation_state)))
                    mutation_result.append((f + (mutation_position, mutation_state)))

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
                 adaptive=True,
                 network=None):
        super(EvolutionMutation, self).__init__(method,
                                                multi_points,
                                                adaptive=adaptive,
                                                max_generation=max_generation,
                                                k0=k0,
                                                k1=k1,
                                                network=network)
    
    def mutate(self, *args, **kwargs):
        population = kwargs['population']
        explore_position = kwargs['explore_position']
        problem = kwargs['problem']
        self.hierarchical = kwargs['hierarchical']

        # traversal_order = list(nx.topological_sort(graph))
        # pos_map = {}
        # for node_name in traversal_order:
        #     cur_node = graph.node[node_name]
        #     pos_map[cur_node['sampling_param']] = node_name

        fitness_values = []
        for individual_index, individual in enumerate(population.population):
            fitness_values.append((individual_index,                        # index
                                   1.0-individual.objectives[0],            # accuracy
                                   individual.features,                     # feature
                                   None))

        mutation_individuals = \
            self.adaptive_mutate(fitness_values=fitness_values,
                                 explore_position=explore_position)
        
        mutation_population = Population()
        for _, individual in enumerate(mutation_individuals):
            if individual[-1] is not None:
                individual_index = individual[0]
                mutation_position = individual[-2]
                mutation_state = individual[-1]

                new_individual = problem.generateIndividual()
                mutated_feature = copy.deepcopy(population.population[individual_index].features)

                for mp, ms in zip(mutation_position, mutation_state):
                    mutated_feature[mp] = ms

                new_individual.features = mutated_feature
                new_individual.objectives = copy.deepcopy(population.population[individual_index].objectives)
                mutation_population.population.append(new_individual)

        return mutation_population
