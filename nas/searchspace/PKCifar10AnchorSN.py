# -*- coding: UTF-8 -*-
# @Time    : 2020-05-01 11:35
# @File    : PKCifar10AnchorSN.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import networkx as nx
from nas.component.NetworkCell import *
from nas.networks.AnchorsUniformSamplingSuperNetwork import *
from nas.searchspace.PKCifar10SN import Cifar10CellBlock
from nas.searchspace.PKCifar10SN import Cifar10ReductionCellBlock
from nas.utils.drawers.NASDrawer import NASDrawer
from nas.component.Loss import *
from nas.component.ClassificationAccuracyEvaluator import *
from nas.searchspace.StageBlockCellArc import *

__all__ = ['PKCifar10AnchorSN']


class PKCifar10AnchorSN(AnchorsUniformSamplingSuperNetwork):
    def __init__(self,
                 blocks_per_stage,
                 cells_per_block,
                 channels_per_block,
                 data_prop,
                 static_proba, *args, **kwargs):
        super(PKCifar10AnchorSN, self).__init__(*args, **kwargs)
        NetworkBlock.state_num = 5
        self.in_chan = data_prop['in_channels']
        self.in_size = data_prop['img_dim']
        self.out_dim = data_prop['out_size'][0]

        self.static_node_proba = static_proba
        self._input_size = (self.in_chan, self.in_size, self.in_size)

        self.blocks = nn.ModuleList([])
        self.graph = nx.DiGraph()
        self.sampling_parameters = nn.ParameterList()
        self._loss = cross_entropy
        self._accuracy_evaluator = ClassificationAccuracyEvaluator()

        # head (固定计算节点，对应激活参数不可学习)
        head = ConvBn(self.in_chan, channels_per_block[0][0], k_size=3, stride=1, relu=True)
        # tail (固定计算节点，结构不可学习)
        tail = kwargs['out_layer']

        # search space（stage - block - cell）
        self.sbca = \
            StageBlockCellArc(Cifar10CellBlock,
                              Cifar10ReductionCellBlock,
                              AddBlock,
                              ConvBn,
                              self.graph,
                              is_cell_dense=True,
                              is_block_dense=True)
        in_name, out_name =\
            self.sbca.generate(head,
                               tail,
                               blocks_per_stage,
                               cells_per_block,
                               channels_per_block)
        self.sampling_parameters = self.sbca.sampling_parameters
        self.blocks = self.sbca.blocks

        # set graph
        self.set_graph(self.graph, in_name, out_name)

    def loss(self, predictions, labels):
        return self._loss(predictions, labels)

    def accuray(self, predictions, labels):
        return self._accuracy_evaluator.accuracy(predictions, labels)

    def hierarchical(self):
        return self.sbca.hierarchical
