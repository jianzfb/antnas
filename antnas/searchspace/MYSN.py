# -*- coding: UTF-8 -*-
# @Time    : 2020/10/27 11:15 上午
# @File    : MYSN.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import networkx as nx
from antnas.component.NetworkCell import *
from antnas.networks.UniformSamplingSuperNetwork import UniformSamplingSuperNetwork
from antnas.networks.AnchorsUniformSamplingSuperNetwork import *
from antnas.networks.EvolutionSuperNetwork import *
from antnas.utils.drawers.NASDrawer import NASDrawer
from antnas.component.Loss import *
from antnas.component.ClassificationAccuracyEvaluator import *
from antnas.searchspace.StageBlockCellArc import *

__all__ = ['MYSN']


class PBlock(NetworkBlock):
    n_layers = 0
    n_comp_steps = 0

    def __init__(self, in_chan=0, out_chan=0, latency=1.0):
        super(PBlock, self).__init__()
        self.structure_fixed = False
        self.latency = latency
        self.params = {
            'module_list': ['PBlock'],
            'name_list': ['PBlock'],
            'PBlock': {},
            'in_chan': in_chan,
            'out_chan': out_chan
        }

    def forward(self, x, sampling=None):
        if sampling is None:
            return x

        is_activate = (int)(sampling.item())
        if is_activate == 1:
            return x
        else:
            return torch.zeros(x.shape, device=x.device)

    def get_flop_cost(self, x):
        return [0]+[self.latency]+[0]*(NetworkBlock.state_num - 2)

    def get_latency(self, x):
        return [[0] + [self.latency] + [0] * (NetworkBlock.state_num - 2),[0] + [self.latency] + [0] * (NetworkBlock.state_num - 2)]


class PBlockCell(NetworkBlock):
    n_layers = 3
    n_comp_steps = 1

    def __init__(self, channles, out_channels):
        super(PBlockCell, self).__init__()
        self.channles = channles
        self.op_list = nn.ModuleList()
        self.params = {
            'module_list': ['PBlock',
                            'PBlock',
                            'PBlock',
                            'PBlock',
                            'PBlock'],
            'name_list': ['a',
                          'b',
                          'c',
                          'd',
                          'e'],
            'a': {'in_chan': channles,
                     'out_chan': out_channels,
                     'latency': 1.0},
            'b': {'in_chan': channles,
                  'out_chan': out_channels,
                  'latency': 2.0},
            'c': {'in_chan': channles,
                  'out_chan': out_channels,
                  'latency': 3.0},
            'd': {'in_chan': channles,
                  'out_chan': out_channels,
                  'latency': 4.0},
            'e': {'in_chan': channles,
                  'out_chan': out_channels,
                  'latency': 5.0},
            'in_chan': channles,
            'out_chan': out_channels
        }
        self.structure_fixed = True

        self.op_list = nn.ModuleList()
        module_list = self.params['module_list']
        name_list = self.params['name_list']
        for block_name, block_module in zip(name_list, module_list):
            block_param = self.params[block_name]
            self.op_list.append(globals()[block_module](**block_param))
        assert(len(self.op_list) == NetworkBlock.state_num)

    def forward(self, input, sampling=None):
        if sampling is None:
            # only for ini
            return self.op_list[0](input)

        op_index = (int)(sampling.item())
        result = self.op_list[op_index](input)
        return result

    def get_flop_cost(self, x):
        cost_list = [0]
        for i in range(len(self.op_list) - 1):
            cost_list.append(self.op_list[i + 1].get_flop_cost(x)[1])

        return cost_list

    def get_latency(self, x):
        cost_list = []
        if NetworkBlock.device_num == 1:
            for i in range(len(self.op_list)):
                cost_list.append(self.op_list[i].get_latency(x)[1])
        else:
            cost_list = [[], []]
            for i in range(len(self.op_list)):
                cost_list[0].append(self.op_list[i].get_latency(x)[0][1])
                cost_list[1].append(self.op_list[i].get_latency(x)[1][1])
        return cost_list

    def get_param_num(self, x):
        cost_list = [0]
        for i in range(len(self.op_list) - 1):
            cost_list.append(self.op_list[i + 1].get_param_num(x)[1])

        return cost_list


class MYSN(UniformSamplingSuperNetwork):
    def __init__(self,
                 blocks_per_stage,
                 cells_per_block,
                 channels_per_block,
                 data_prop,
                 *args, **kwargs):
        super(MYSN, self).__init__(*args, **kwargs)
        NetworkBlock.state_num = 5
        self.in_chan = data_prop['in_channels']
        self.in_size = data_prop['img_dim']
        self.out_dim = data_prop['out_size'][0]

        self._input_size = (self.in_chan, self.in_size, self.in_size)
        self.graph = nx.DiGraph()
        self.sampling_parameters = nn.ParameterList()
        self.blocks = nn.ModuleList([])
        self._loss = nn.CrossEntropyLoss()
        head = PBlock(10, 10)
        self.graph.add_node(SuperNetwork._INPUT_NODE_FORMAT.format(0, 0),
                            module=len(self.blocks),
                            module_params=head.params,
                            sampling_param=len(self.blocks),
                            structure_fixed=head.structure_fixed,
                            pos=(0, 0))
        self.blocks.append(head)

        ################## level - 1 ########################
        left_1 = PBlockCell(10, 10)
        self.graph.add_node(SuperNetwork._CELL_NODE_FORMAT.format(0, 1),
                            module=len(self.blocks),
                            module_params=left_1.params,
                            sampling_param=len(self.blocks),
                            structure_fixed=left_1.structure_fixed,
                            pos=(0, 1))
        self.blocks.append(left_1)
        self.graph.add_edge(SuperNetwork._INPUT_NODE_FORMAT.format(0, 0),
                            SuperNetwork._CELL_NODE_FORMAT.format(0, 1),
                            width_node=SuperNetwork._CELL_NODE_FORMAT.format(0, 1))

        right_1 = PBlockCell(10, 10)
        self.graph.add_node(SuperNetwork._CELL_NODE_FORMAT.format(1, 1),
                            module=len(self.blocks),
                            module_params=right_1.params,
                            sampling_param=len(self.blocks),
                            structure_fixed=right_1.structure_fixed,
                            pos=(1, 1))
        self.blocks.append(right_1)
        self.graph.add_edge(SuperNetwork._INPUT_NODE_FORMAT.format(0, 0),
                            SuperNetwork._CELL_NODE_FORMAT.format(1, 1),
                            width_node=SuperNetwork._CELL_NODE_FORMAT.format(1, 1))
        ################## end ########################

        ################## level - 2 ########################
        left_2 = PBlockCell(10,10)
        self.graph.add_node(SuperNetwork._CELL_NODE_FORMAT.format(0, 2),
                            module=len(self.blocks),
                            module_params=left_2.params,
                            sampling_param=len(self.blocks),
                            structure_fixed=left_2.structure_fixed,
                            pos=(0, 2))
        self.blocks.append(left_2)
        self.graph.add_edge(SuperNetwork._CELL_NODE_FORMAT.format(0, 1),
                            SuperNetwork._CELL_NODE_FORMAT.format(0, 2),
                            width_node=SuperNetwork._CELL_NODE_FORMAT.format(0, 2))

        right_2 = PBlockCell(10, 10)
        self.graph.add_node(SuperNetwork._CELL_NODE_FORMAT.format(1, 2),
                            module=len(self.blocks),
                            module_params=right_2.params,
                            sampling_param=len(self.blocks),
                            structure_fixed=right_2.structure_fixed,
                            pos=(1, 2))
        self.blocks.append(right_2)
        self.graph.add_edge(SuperNetwork._CELL_NODE_FORMAT.format(1, 1),
                            SuperNetwork._CELL_NODE_FORMAT.format(1, 2),
                            width_node=SuperNetwork._CELL_NODE_FORMAT.format(1, 2))
        ################## end ##############################
        merge = AddBlock()
        self.graph.add_node(SuperNetwork._AGGREGATION_NODE_FORMAT.format(0, 3),
                            module=len(self.blocks),
                            module_params=merge.params,
                            sampling_param=len(self.blocks),
                            structure_fixed=merge.structure_fixed,
                            pos=(0, 3))
        self.blocks.append(merge)
        self.graph.add_edge(SuperNetwork._CELL_NODE_FORMAT.format(0, 2),
                            SuperNetwork._AGGREGATION_NODE_FORMAT.format(0, 3),
                            width_node=SuperNetwork._AGGREGATION_NODE_FORMAT.format(0, 3))
        self.graph.add_edge(SuperNetwork._CELL_NODE_FORMAT.format(1, 2),
                            SuperNetwork._AGGREGATION_NODE_FORMAT.format(0, 3),
                            width_node=SuperNetwork._AGGREGATION_NODE_FORMAT.format(0, 3))

        tail = kwargs['out_layer']
        self.graph.add_node(SuperNetwork._OUTPUT_NODE_FORMAT.format(0, 4),
                            module=len(self.blocks),
                            module_params=tail.params,
                            sampling_param=len(self.blocks),
                            structure_fixed=tail.structure_fixed,
                            pos=(0, 4))
        self.blocks.append(tail)
        self.graph.add_edge(SuperNetwork._AGGREGATION_NODE_FORMAT.format(0, 3),
                            SuperNetwork._OUTPUT_NODE_FORMAT.format(0, 4),
                            width_node=SuperNetwork._OUTPUT_NODE_FORMAT.format(0, 4))

        # set graph
        in_name = SuperNetwork._INPUT_NODE_FORMAT.format(0, 0)
        out_name = SuperNetwork._OUTPUT_NODE_FORMAT.format(0, 4)
        self.set_graph(self.graph, in_name, out_name)

        # 保存搜索空间图
        a = NASDrawer()
        a.draw(self.graph, filename='./searchspace.svg')

    def loss(self, predictions, labels):
        return self._loss(predictions, labels)

    def accuracy_evaluator(self):
        return ClassificationAccuracyEvaluator()

    def hierarchical(self):
        return self.sbca.hierarchical
