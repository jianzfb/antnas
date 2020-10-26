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
from antnas.networks.nsga2 import *
from antnas.component.NetworkBlock import *


def refine_device_select(devices, net):
    # 设备选择约束条件
    # CELL 节点：自由选择设备
    # A,T,L节点：设备选择保持与下继节点设备选择一致
    # I,O,F节点：设备保持不变（0）
    traversal_order = list(nx.topological_sort(net))
    for node_name in traversal_order:
        cur_node = net.node[node_name]
        if node_name.startswith('I') or node_name.startswith('O') or node_name.startswith('FIXED'):
            devices[cur_node['sampling_param']] = 0

    for node_name in traversal_order:
        cur_node = net.node[node_name]
        if node_name.startswith('A'):
            # 寻找下继节点
            # 按照构建搜索空间的约定，A节点的后继节点必然是CELL节点或O节点
            for nn in net.successors(node_name):
                nn_node = net.node[nn]
                assert (nn.startswith('CELL') or nn.startswith('O'))
                devices[cur_node['sampling_param']] = devices[nn_node['sampling_param']]
                break
        elif node_name.startswith('T'):
            # 寻找上继节点
            # 按照构建搜索空间的约定，T节点的上继节点必然CELL节点
            for pre in net.predecessors(node_name):
                assert (pre.startswith('CELL'))
                pre_node = net.node[pre]
                devices[cur_node['sampling_param']] = devices[pre_node['sampling_param']]
                break
    return devices


class Mutation(object):
    def __init__(self, multi_points, **kwargs):
        self.multi_points = multi_points    # -1: auto
        self._generation = 0
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

    def _mutate_device_based_matrices(self, *args, **kwargs):
        fitness_values = kwargs['fitness_values']       # 设备

        # fitness_values: [(index, fitness, gene), (), ]
        N = len(fitness_values)         # 个数
        M = len(fitness_values[0][2])   # 基因长度

        C = np.zeros((N, 1))  # fitness cumulative probability of chromosome i,
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
        alpha = 1.0 - C  # the probability to choose which individual for mutation

        # 选择允许进行设备变异位置（仅考虑CELL节点）
        explore_position = []
        for node_name in list(nx.topological_sort(self.network.net)):
            cur_node = self.network.net.node[node_name]
            if node_name.startswith('CELL'):
                explore_position.append(cur_node['sampling_param'])

        # 变异选择
        mutation_result = []
        for f_index, f in enumerate(fitness_values):
            # 变异比例
            mutation_ratio = alpha[f[0]]

            # 2.step mutation pos selection
            # mutation points number
            multi_points = self.multi_points if self.multi_points > 0 else int(mutation_ratio * len(explore_position))
            multi_points = min(multi_points, len(explore_position))
            if multi_points == 0:
                print('(device select) no mutation pos')
                mutation_result.append((f + ([], [])))
            else:
                mutation_position = np.random.choice(explore_position, multi_points, replace=False)
                mutation_position = mutation_position.flatten().tolist()
                mutation_state = []
                for mutation_p in mutation_position:
                    candidate_state = set(list(range(NetworkBlock.device_num)))
                    candidate_state.discard(f[2][mutation_p])
                    candidate_state = list(candidate_state)

                    s = np.random.choice(candidate_state,
                                         1,
                                         replace=False)
                    mutation_state.append(int(s))

                print("(device select) individual %d mutation at %s to state %s" % (f_index, str(mutation_position), str(mutation_state)))
                mutation_result.append((f + (mutation_position, mutation_state)))

        return mutation_result

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
            if multi_points == 0:
                print('no mutation pos')
                mutation_result.append((f + ([], [])))
            else:
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
                                if s not in [0, 1]:
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


class EvolutionMutation(Mutation):
    def __init__(self,
                 multi_points,
                 network=None):
        super(EvolutionMutation, self).__init__(multi_points, network=network)
    
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

        # 1.step 网络结构及算子 变异
        fitness_values = []
        for individual_index, individual in enumerate(population.population):
            fitness_values.append((individual_index,                        # index
                                   1.0-individual.objectives[0],            # accuracy
                                   individual.features,                     # feature
                                   None))

        mutation_individuals = \
            self._mutate_based_matrices(fitness_values=fitness_values,
                                        explore_position=explore_position)

        mutation_population = Population()
        index_map = {}
        for _, individual in enumerate(mutation_individuals):
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
            # 记录变异个体和原个体映射关系
            index_map[individual_index] = len(mutation_population.population) - 1

        # 2.step 计算设备 变异
        if population.population[0].devices is not None:
            fitness_values = []
            for individual_index, individual in enumerate(population.population):
                fitness_values.append((individual_index,                                # index
                                       1.0/(individual.objectives[1]+0.000000001),      # architecture loss
                                       individual.devices,                              # feature
                                        ))

            mutation_individuals = \
                self._mutate_device_based_matrices(fitness_values=fitness_values)

            for _, individual in enumerate(mutation_individuals):
                individual_index = individual[0]
                mutation_position = individual[-2]
                mutation_state = individual[-1]

                mutation_population.population[index_map[individual_index]].devices = \
                    copy.deepcopy(population.population[individual_index].devices)

                # 设置变异后的节点设备
                for mp, ms in zip(mutation_position, mutation_state):
                    mutation_population.population[index_map[individual_index]].devices[mp] = ms

                # 重新约束个体的设备选择有效性
                mutation_population.population[index_map[individual_index]].devices = \
                    refine_device_select(mutation_population.population[index_map[individual_index]].devices,
                                         self.network.net)

        return mutation_population
