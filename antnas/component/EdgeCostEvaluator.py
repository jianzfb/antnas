# -*- coding: UTF-8 -*-
# @Time    : 2019-10-29 12:15
# @File    : EdgeCostEvaluator.py
# @Author  : jian<jian@mltalker.com>
from antnas.component.CostEvaluator import CostEvaluator
from antnas.component.NetworkBlock import *
import torch
import networkx as nx
from ctypes import cdll
import ctypes
heft_dll = cdll.LoadLibrary('./antnas/extent/heft.so')
heft_dll.get.restype = ctypes.c_double


class EdgeCostEvaluator(CostEvaluator):
    base_transfer_time = 2.0

    def get_cost(self, architectures, devices=None):
        # self.costs is N x state or N x state x device
        # 结构代价
        # 如果指定多设备，则重置architectures
        if devices is None:
            init_costs = self.costs
            if len(self.costs.shape) == 3:
                init_costs = self.costs[:, 0, :]
            costs = torch.gather(init_costs, dim=1, index=architectures.long())
            architecture_cost = costs.sum(0)
            return architecture_cost
        else:
            assert(len(self.costs.shape) == 3)
            # devices_num = len(devices)
            devices_num = 2
            task_num = len(self.model.traversal_order)

            computing_cost = [0 for _ in range(task_num*devices_num)]
            communication_cost = [-1 for _ in range(task_num*task_num)]

            for node in self.model.traversal_order:
                from_index = self.model.arch_node_index[node]
                for succ in self.model.net.successors(node):
                    to_index = self.model.arch_node_index[succ]
                    communication_cost[(int)(from_index * task_num + to_index)] = \
                        EdgeCostEvaluator.base_transfer_time

                    if node.startswith('T'):
                        if (int)(architectures[from_index, 0]) == 0:
                            communication_cost[(int)(from_index * task_num + to_index)] = 0

            for node in self.model.traversal_order:
                from_index = self.model.arch_node_index[node]
                for device_index in range(devices_num):
                    computing_cost[from_index*devices_num+device_index] =\
                        (float)(self.costs[from_index, (int)(device_index), (int)(architectures[from_index, 0])])

            ccomputing_cost = (ctypes.c_double * len(computing_cost))(*computing_cost)
            ccommunication_cost = (ctypes.c_double * len(communication_cost))(*communication_cost)
            time_span = heft_dll.get(task_num, devices_num, ccomputing_cost, ccommunication_cost)
            architecture_cost = torch.tensor(time_span)

            return torch.reshape(architecture_cost, [1])

    @property
    def total_cost(self):
        return self.costs.sum().item()
