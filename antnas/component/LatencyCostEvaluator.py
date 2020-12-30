# -*- coding: UTF-8 -*-
# @Time    : 2019-10-29 12:15
# @File    : LatencyCostEvaluator.py
# @Author  : jian<jian@mltalker.com>
import logging
import torch
from antnas.component.EdgeCostEvaluator import EdgeCostEvaluator
from antnas.component.NetworkBlock import NetworkBlock
logger = logging.getLogger(__name__)


class LatencyCostEvaluator(EdgeCostEvaluator):
    def __init__(self, *args, **kwargs):
        super(LatencyCostEvaluator, self).__init__(*args, **kwargs)
        # load latency lookup table
        if 'latency' in self.kwargs:
            NetworkBlock.load_lookup_table(self.kwargs['latency'])
            print('load latency lookup table')
        else:
            print('latency lookup table dont exist')

    def init_costs(self, model, graph, is_main_cost=False, input_node=None, input_shape=None):
        print('initialize latency cost')
        # 对于多设备来说，需要获得每个设备下的每个节点不同算子下的时间代价
        with torch.no_grad():
            # set costs
            costs = None
            if NetworkBlock.device_num == 1:
                costs = torch.Tensor(graph.number_of_nodes(), NetworkBlock.state_num)
            else:
                costs = torch.Tensor(graph.number_of_nodes(), NetworkBlock.device_num, NetworkBlock.state_num)

            if input_node is not None and input_shape is not None:
                self.input_node = input_node
                self.input_shape = input_shape

            data_dict = {}
            data_dict[self.input_node] = [torch.ones(self.input_shape)]

            for node in model.traversal_order:
                cur_node = graph.node[node]
                # input = graph.node[node]['input']
                input = data_dict[node]

                if (isinstance(input, tuple) or isinstance(input, list)) and len(input) == 1:
                    input = input[0]

                # 获得输出
                # print(node)
                out = model.blocks[cur_node['module']](input)
                # 获得代价
                if node.startswith('F'):
                    cost = [0.0] * NetworkBlock.state_num
                else:
                    cost = model.blocks[cur_node['module']].get_latency(input)

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
