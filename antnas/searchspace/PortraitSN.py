# @Time    : 2020/5/29 10:56
# @Author  : zhangchenming
import torch.nn as nn
import networkx as nx
from antnas.networks.UniformSamplingSuperNetwork import UniformSamplingSuperNetwork
from antnas.component.Loss import cross_entropy
from antnas.component.SegmentationAccuracyEvaluator import SegmentationAccuracyEvaluator
from antnas.searchspace.FixSegArc import FixSegArc


class PortraitSN(UniformSamplingSuperNetwork):
    def __init__(self, data_prop, *args, **kwargs):
        super(PortraitSN, self).__init__(*args, **kwargs)

        self.in_chan = data_prop['in_channels']
        self.in_size = data_prop['img_dim']
        self.out_dim = data_prop['out_size'][0]

        self.graph = nx.DiGraph()
        self.sampling_parameters = nn.ParameterList()

        self._loss = cross_entropy
        self._accuracy_evaluator = SegmentationAccuracyEvaluator(class_num=self.out_dim)

        self.seg_arc = FixSegArc(self.in_chan, self.graph)
        in_name, out_name = self.seg_arc.generate()
        self.blocks = self.seg_arc.blocks
        self.set_graph(self.graph, in_name, out_name)

    def loss(self, predictions, labels):
        return self._loss(predictions, labels)

    def accuray(self, predictions, labels):
        return self._accuracy_evaluator.accuracy(predictions, labels)

    def hierarchical(self):
        return self.seg_arc.hierarchical