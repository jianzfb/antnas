# -*- coding: UTF-8 -*-
# @Time    : 2020-04-03 18:05
# @File    : PKArc.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from nas.networks.SuperNetwork import *
from nas.utils.drawers.BSNDrawer import BSNDrawer
import torch
import torch.nn as nn


class PKArc:
    def __init__(self, graph):
        self.graph = graph
        self.blocks = nn.ModuleList([])
        self.sampling_parameters = None
        self.node_map = {}
        self.inv_node_map = {}

    def save(self, folder, name):
        architecture_path = os.path.join(folder, 'pk_%s.architecture'%name)
        nx.write_gpickle(self.graph, architecture_path)

    def add(self, module, name):
        pos = (0, len(self.blocks))
        pos_name = SuperNetwork._FIXED_NODE_FORMAT.format(*pos)
        self.graph.add_node(pos_name,
                            module=len(self.blocks),
                            module_params=module.params,
                            sampling_param=len(self.blocks),
                            pos=BSNDrawer.get_draw_pos(pos=pos),
                            sampled=1)
        self.node_map[name] = pos_name
        self.inv_node_map[pos_name] = name
        self.blocks.append(module)

    def link(self, from_name, to_name):
        if from_name not in self.node_map:
            print('%s node not in graph'%from_name)
            return

        if to_name not in self.node_map:
            print('%s node not in graph'%to_name)
            return

        self.graph.add_edge(self.node_map[from_name],self.node_map[to_name], width_node=self.node_map[to_name])

    def generate(self, head, tail):
        traversal_order = list(nx.topological_sort(self.graph))
        front_in_graph = traversal_order[0]
        end_in_graph = traversal_order[-1]
        in_name = front_in_graph
        if head is not None:
            # link head to arc
            pos = (0, len(self.blocks))
            in_name = SuperNetwork._INPUT_NODE_FORMAT.format(*pos)
            self.graph.add_node(in_name,
                                module=len(self.blocks),
                                module_params=head.params,
                                sampling_param=len(self.blocks),
                                pos=BSNDrawer.get_draw_pos(pos=pos),
                                sampled=1)
            self.blocks.append(head)

            self.graph.add_edge(in_name,
                                front_in_graph,
                                width_node=front_in_graph)

        # link arc to tail
        pos = (0, len(self.blocks))
        out_name = SuperNetwork._OUTPUT_NODE_FORMAT.format(*pos)
        self.graph.add_node(out_name,
                            module=len(self.blocks),
                            module_params=tail.params,
                            sampling_param=len(self.blocks),
                            pos=BSNDrawer.get_draw_pos(pos=pos),
                            sampled=1)
        self.blocks.append(tail)
        self.graph.add_edge(end_in_graph,
                            out_name,
                            width_node=out_name)

        # TODO Allow several input and/or output nodes
        traversal_order = list(nx.topological_sort(self.graph))
        if traversal_order[0] != in_name or traversal_order[-1] != out_name:
            raise ValueError('Seems like the given graph is broken')

        return in_name, out_name