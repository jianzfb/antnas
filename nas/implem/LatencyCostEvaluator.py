# -*- coding: UTF-8 -*-
# @Time    : 2019-10-29 12:15
# @File    : LatencyCostEvaluator.py
# @Author  : jian<jian@mltalker.com>
import logging
import torch
from nas.implem.EdgeCostEvaluator import EdgeCostEvaluator
from nas.interfaces.NetworkBlock import NetworkBlock
logger = logging.getLogger(__name__)


class LatencyCostEvaluator(EdgeCostEvaluator):
    def __init__(self, *args, **kwargs):
        super(LatencyCostEvaluator, self).__init__(*args, **kwargs)
        # load latency lookup table
        NetworkBlock.load_lookup_table(self.kwargs['latency'])
        print('load latency lookup table')

    def init_costs(self, model, main_cost, graph):
        print('initialize latency cost')
        with torch.no_grad():
            # set costs
            costs = torch.Tensor(graph.number_of_nodes(), NetworkBlock.state_num)

            for node in model.traversal_order:
                cur_node = graph.node[node]
                input = graph.node[node]['input']
                if (isinstance(input, tuple) or isinstance(input, list)) and len(input) == 1:
                    input = input[0]

                print('latency node %s'%node)
                if node.startswith('F'):
                    cost = [0.0] * NetworkBlock.state_num
                else:
                    cost = model.blocks[cur_node['module']].get_latency(input)

                if main_cost:
                    cur_node['cost'] = cost

                costs[self.model.architecture_node_index[node]] = torch.Tensor(cost)

            return costs
