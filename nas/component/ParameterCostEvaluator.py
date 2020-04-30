import logging

import torch

from nas.component.EdgeCostEvaluator import EdgeCostEvaluator
from nas.component.NetworkBlock import NetworkBlock

logger = logging.getLogger(__name__)


class ParameterCostEvaluator(EdgeCostEvaluator):

    def init_costs(self, model, graph, is_main_cost=False):
        with torch.no_grad():
            self.costs = torch.Tensor(graph.number_of_nodes(), NetworkBlock.state_num)

            for node in model.traversal_order:
                cur_node = graph.node[node]
                input = graph.node[node]['input']

                if (isinstance(input, tuple) or isinstance(input, list)) and len(input) == 1:
                    input = input[0]

                cost = model.blocks[cur_node['module']].get_param_num(input)

                if is_main_cost:
                    cur_node['cost'] = cost

                self.costs[self.model.arch_node_index[node]] = torch.Tensor(cost)
                cur_node['input'] = []

            return self.costs