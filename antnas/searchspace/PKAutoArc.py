# -*- coding: UTF-8 -*-
# @Time    : 2020-04-01 11:47
# @File    : PKAutoArc.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antnas.networks.SuperNetwork import *
from antnas.utils.drawers.NASDrawer import NASDrawer
import torch
import torch.nn as nn
from antnas.searchspace.Arc import *


class PKAutoArc(Arc):
    def __init__(self, graph):
        super(PKAutoArc, self).__init__(graph)
        self._names = []

    def generate(self, head, tail, modules):
        in_name = SuperNetwork._INPUT_NODE_FORMAT.format(0, 0)
        self.graph.add_node(in_name,
                            module=len(self.blocks),
                            module_params=head.params,
                            sampling_param=len(self.blocks),
                            structure_fixed=head.structure_fixed,
                            pos=NASDrawer.get_draw_pos(pos=(0, 0)),
                            sampled=1)
        self.blocks.append(head)

        for module_index, module in enumerate(modules):
            module_index += 1

            module_name = SuperNetwork._FIXED_NODE_FORMAT.format(0, module_index)
            # push to _names
            self._names.append(module_name)
            
            pos = (0, module_index)
            self.graph.add_node(module_name,
                                module=len(self.blocks),
                                module_params=module.params,
                                sampling_param=len(self.blocks),
                                structure_fixed=module.structure_fixed,
                                pos=NASDrawer.get_draw_pos(pos=pos),
                                sampled=1)
            self.blocks.append(module)

            if module_index > 1:
                # 固定连接
                self.graph.add_edge(SuperNetwork._FIXED_NODE_FORMAT.format(0, module_index-1),
                                    SuperNetwork._FIXED_NODE_FORMAT.format(0, module_index),
                                    width_node=SuperNetwork._FIXED_NODE_FORMAT.format(0, module_index))
            
        # link head to arc
        self.graph.add_edge(SuperNetwork._INPUT_NODE_FORMAT.format(0, 0),
                            SuperNetwork._FIXED_NODE_FORMAT.format(0, 1),
                            width_node=SuperNetwork._FIXED_NODE_FORMAT.format(0, 1))
        
        if tail is None:
            self.offset = len(modules) + 1
            return in_name, ""
        
        # link arc to tail
        out_name = SuperNetwork._OUTPUT_NODE_FORMAT.format(0, len(modules) + 1)
        self.graph.add_node(out_name,
                            module=len(self.blocks),
                            module_params=tail.params,
                            sampling_param=len(self.blocks),
                            structure_fixed=tail.structure_fixed,
                            pos=NASDrawer.get_draw_pos(pos=(0, len(modules) + 1)),
                            sampled=1)
        self.blocks.append(tail)

        self.graph.add_edge(SuperNetwork._FIXED_NODE_FORMAT.format(0, len(modules)),
                            SuperNetwork._OUTPUT_NODE_FORMAT.format(0, len(modules)+1),
                            width_node=SuperNetwork._OUTPUT_NODE_FORMAT.format(0, len(modules)+1))
        
        self.offset = len(modules) + 2
        
        # TODO Allow several input and/or output nodes
        traversal_order = list(nx.topological_sort(self.graph))
        if traversal_order[0] != in_name or traversal_order[-1] != out_name:
            raise ValueError('Seems like the given graph is broken')

        self.in_node = in_name
        self.out_node = out_name
        return in_name, out_name

    def save(self, folder, name):
        architecture_path = os.path.join(folder, 'pk_%s.architecture'%name)
        nx.write_gpickle(self.graph, architecture_path)
    
    @property
    def names(self):
        return self._names