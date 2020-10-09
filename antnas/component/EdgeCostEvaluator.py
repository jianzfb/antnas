# -*- coding: UTF-8 -*-
# @Time    : 2019-10-29 12:15
# @File    : EdgeCostEvaluator.py
# @Author  : jian<jian@mltalker.com>
from antnas.component.CostEvaluator import CostEvaluator
from antnas.component.NetworkBlock import *
import torch


class EdgeCostEvaluator(CostEvaluator):
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
            devices = torch.as_tensor(devices)
            reshape_costs = torch.reshape(self.costs, (self.costs.shape[0], -1))
            reset_architectures = torch.reshape(devices, (self.costs.shape[0], 1)) * NetworkBlock.state_num + architectures
            costs = torch.gather(reshape_costs, dim=1, index=reset_architectures.long())
            device_num = NetworkBlock.device_num
            node_num = architectures.shape[0]

            # 1. 结构代价
            # devices: n_nodes
            accumulate_cost = torch.zeros((node_num))
            for node_name in self.model.traversal_order:
                node_index = self.model.path_recorder.node_index[node_name]
                incoming = torch.zeros((device_num, node_num))
                dependent_pre_indexes = []
                dependent_pre_devices = []
                for prev in self.model.net.predecessors(node_name):
                    pre_index = self.model.path_recorder.node_index[prev]
                    dependent_pre_indexes.append(pre_index)
                    dependent_pre_devices.append(devices[pre_index])
                    for device_i in range(device_num):
                        incoming[device_i][pre_index] = \
                            (devices[pre_index] == device_i).float() * accumulate_cost[pre_index]

                if len(dependent_pre_indexes) == 0:
                    continue

                # 数据传输代价
                for pre_device in dependent_pre_devices:
                    if pre_device != devices[node_index]:
                        pass

                # device_num x node_num
                incoming = incoming[:, dependent_pre_indexes]
                incoming_min = incoming.min(1, keepdim=True)[0]
                incoming = incoming - incoming_min
                incoming = torch.sum(incoming, 1, keepdim=True)
                incoming = incoming + incoming_min
                # device_num(1) x node_num(1)
                accumulate_cost[node_index] = \
                    costs[node_index] + incoming.max(0, keepdim=True)[0].squeeze()

            out_index = self.model.path_recorder.node_index[self.model.out_node]
            architecture_cost = accumulate_cost[out_index]

            return architecture_cost

    @property
    def total_cost(self):
        return self.costs.sum().item()
