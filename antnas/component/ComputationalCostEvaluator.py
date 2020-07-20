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
    def init_costs(self, model, graph, is_main_cost=False):
        print('initialize computation cost')
        with torch.no_grad():
            # set costs
            costs = torch.Tensor(graph.number_of_nodes(), NetworkBlock.state_num)

            for node in model.traversal_order:
                cur_node = graph.node[node]
                # if isinstance(cur_node['module'], NetworkBlock):
                #     cost = cur_node['module'].get_flop_cost()
                # else:
                #     raise RuntimeError("Deprecated behaviour, all module should now be a NetworkBlock, got {}".format(
                #         type(cur_node['module']).__name__))
                input = graph.node[node]['input']
                if (isinstance(input, tuple) or isinstance(input, list)) and len(input) == 1:
                    input = input[0]

                cost = model.blocks[cur_node['module']].get_flop_cost(input)

                if is_main_cost:
                    cur_node['cost'] = cost

                costs[self.model.arch_node_index[node]] = torch.Tensor(cost)

            self.costs = costs
            return costs
