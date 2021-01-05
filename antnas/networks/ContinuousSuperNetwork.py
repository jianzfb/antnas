# -*- coding: UTF-8 -*-
# @Time    : 2020/12/24 11:16 上午
# @File    : ContinuousSuperNetwork.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from antnas.networks.SuperNetwork import SuperNetwork
from antnas.component.NetworkBlock import *
from antnas.component.NetworkCell import *
from antnas.component.PathRecorder import PathRecorder
from antnas.component.NetworkBlock import *

import copy
import networkx as nx
from antnas.networks.UniformSamplingSuperNetwork import *
import threading


class ContinuousSuperNetwork(UniformSamplingSuperNetwork):
    lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        super(ContinuousSuperNetwork, self).__init__(*args, **kwargs)
        self.candidate_archs = []
        self.explore_position = []
        self.pos_map = {}
        self.using_static_arch = False
        self.disturb_ratio = 0.1

    def update(self, *args, **kwargs):
        # 使用精英种群更新候选结构池
        elited_population = kwargs.get('elited_population', None)
        if elited_population is None:
            return

        elited_archs = []
        for individual in elited_population:
            elited_archs.append(individual.features)

        # 加锁
        ContinuousSuperNetwork.lock.acquire()
        self.candidate_archs = elited_archs
        ContinuousSuperNetwork.lock.release()

    def init(self, *args, **kwargs):
        super(ContinuousSuperNetwork, self).init(*args, **kwargs)

        # 加载初始候选结构
        feature = [None for _ in range(len(self.traversal_order))]
        for node_index, node_name in enumerate(self.traversal_order):
            cur_node = self.graph.node[node_name]
            sampled = getattr(cur_node, 'sampled', None)
            if sampled is None:
                sampled = 1
            feature[cur_node['sampling_param']] = sampled

        assert(feature[0] is not None)
        self.candidate_archs = [feature]

        # 初始化候选结构
        for node_name in self.traversal_order:
            cur_node = self.net.node[node_name]
            if node_name.startswith('CELL') or node_name.startswith('T'):
                self.explore_position.append(cur_node['sampling_param'])

        for node_name in self.traversal_order:
            cur_node = self.net.node[node_name]
            self.pos_map[cur_node['sampling_param']] = node_name

    def sample_arch(self, *args, **kwargs):
        if self.using_static_arch:
            feature = [None for _ in range(len(self.traversal_order))]
            for node_index, node_name in enumerate(self.traversal_order):
                cur_node = self.graph.node[node_name]
                feature[cur_node['sampling_param']] = cur_node['sampled']
            return feature

        # 加锁
        ContinuousSuperNetwork.lock.acquire()

        # 1.step 从候选池中，随机选择结构
        random_i = np.random.choice(list(range(len(self.candidate_archs))), 1)
        feature = copy.deepcopy(self.candidate_archs[(int)(random_i)])

        # 2.step 随机挑选10%个节点进行
        disturb_num = (int)(self.disturb_ratio * len(self.explore_position))
        disturb_position = np.random.choice(self.explore_position, disturb_num, replace=False)
        if type(disturb_position) != list:
            disturb_position = disturb_position.flatten().tolist()

        for p in disturb_position:
            node_name = self.pos_map[p]
            node = self.net.node[node_name]

            if not (node_name.startswith('CELL') or node_name.startswith('T')):
                # 不可学习，处于永远激活状态
                feature[p] = int(1)
            else:
                if not self.blocks[node['module']].structure_fixed:
                    if feature[p] == 1:
                        feature[p] = 0
                    else:
                        feature[p] = 1
                else:
                    while True:
                        s = int(np.random.randint(0, NetworkBlock.state_num))
                        if feature[p] != s:
                            feature[p] = s
                            break

        ContinuousSuperNetwork.lock.release()
        return feature
