# -*- coding: UTF-8 -*-
# @Time    : 2019-10-29 12:15
# @File    : ComputationalCostEvaluator.py
# @Author  : jian<jian@mltalker.com>
import logging
import torch
from antnas.component.EdgeCostEvaluator import EdgeCostEvaluator
from antnas.component.NetworkBlock import NetworkBlock

logger = logging.getLogger(__name__)


class ComputationalCostEvaluator(EdgeCostEvaluator):
    def init_costs(self, model, graph, is_main_cost=False, input_node=None, input_shape=None):
        print('initialize computation cost')
        with torch.no_grad():
            # set costs
            costs = torch.Tensor(graph.number_of_nodes(), NetworkBlock.state_num)

            if input_node is not None and input_shape is not None:
                self.input_node = input_node
                self.input_shape = input_shape

            data_dict = {}
            data_dict[self.input_node] = [torch.ones(self.input_shape)]

            for node in model.traversal_order:
                cur_node = graph.node[node]
                input = data_dict[node]

                if (isinstance(input, tuple) or isinstance(input, list)) and len(input) == 1:
                    input = input[0]

                # 获得输出
                out = model.blocks[cur_node['module']](input)
                # 获得代价
                cost = model.blocks[cur_node['module']].get_flop_cost(input)

                if is_main_cost:
                    cur_node['cost'] = cost

                costs[self.model.arch_node_index[node]] = torch.Tensor(cost)

                # set successor input
                for succ in graph.successors(node):
                    if succ not in data_dict:
                        data_dict[succ] = []

                    data_dict[succ].append(out)

            self.costs = costs
            return costs
