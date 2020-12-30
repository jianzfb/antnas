# -*- coding: UTF-8 -*-
# @Time    : 2020/12/24 5:24 下午
# @File    : PKHRNetSN.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import networkx as nx
from antnas.component.NetworkCell import *
from antnas.networks.ContinuousSuperNetwork import ContinuousSuperNetwork
from antnas.utils.drawers.NASDrawer import NASDrawer
from antnas.component.ClassificationAccuracyEvaluator import *
from antnas.component.SegmentationAccuracyEvaluator import *
from antnas.searchspace.GridArc import *
from antnas.component.Loss import *


__all__ = ['PKHRNetSN']


class HRCellBlock(NetworkBlock):
    expansion = 1

    def __init__(self, in_chan, out_chan):
        super(HRCellBlock, self).__init__()
        self.op_list = nn.ModuleList()

        # for basic block
        self.params = {
            'module_list': [
                'Skip',
                'BasicBlock'],
            'name_list': [
                'Skip',
                'BasicBlock_k3'],
            'Skip': {
                'in_chan': in_chan,
                'out_chan': out_chan,
                'reduction': False},
            'BasicBlock_k3': {
                'in_chan': in_chan,
                'out_chan': out_chan,
                'k_size': 3},
            'in_chan': in_chan,
            'out_chan': out_chan
        }

        # # for bottleneck block
        # self.params = {
        #     'module_list': [
        #         'Skip',
        #         'BottleneckBlock'],
        #     'name_list': [
        #         'Skip',
        #         'BottleneckBlock_k3'],
        #     'Skip': {
        #         'in_chan': in_chan if in_chan != out_chan else in_chan * self.expansion,
        #         'out_chan': out_chan*self.expansion,
        #         'reduction': False},
        #     'BottleneckBlock_k3': {
        #         'in_chan': in_chan if in_chan != out_chan else in_chan * self.expansion,
        #         'out_chan': out_chan,
        #         'expansion': self.expansion,
        #         'k_size': 3},
        #     'in_chan': in_chan,
        #     'out_chan': out_chan
        # }
        self.op_list = self.build()
        self.structure_fixed = True
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


# 同分辨率转换
class SameTransformer(NetworkBlock):
    def __init__(self, in_chan, out_chan):
        super(SameTransformer, self).__init__()
        self.convbn = None
        if in_chan != out_chan:
            self.convbn = \
                ConvBn(in_chan*HRCellBlock.expansion,
                       out_chan*HRCellBlock.expansion,
                       relu=True,
                       k_size=3)
        self.structure_fixed = False

        self.params = {
            'module_list': ['SameTransformer'],
            'name_list': ['SameTransformer'],
            'SameTransformer': {'out_chan': out_chan,
                                    'in_chan': in_chan},
            'in_chan': in_chan,
            'out_chan': out_chan
        }

    def forward(self, input, sampling=None):
        y = input
        if self.convbn is not None:
            y = self.convbn(y)

        if sampling is None:
            # only for ini
            return y

        is_activate = (int)(sampling.item())
        if is_activate == 1:
            return y
        else:
            return torch.zeros(y.shape, device=y.device)


# 从低分辨率到高分辨率转换
class UpsampleTransformer(NetworkBlock):
    def __init__(self, in_chan, out_chan, scale):
        super(UpsampleTransformer, self).__init__()
        self.convbn = ConvBn(in_chan*HRCellBlock.expansion,
                             out_chan*HRCellBlock.expansion,
                             relu=False,
                             k_size=1)
        self.resize = ResizedBlock(0, 0, scale_factor=scale)
        self.structure_fixed = False

        self.params = {
            'module_list': ['UpsampleTransformer'],
            'name_list': ['UpsampleTransformer'],
            'UpsampleTransformer': {'out_chan': out_chan,
                                    'in_chan': in_chan,
                                    'scale': scale},
            'in_chan': in_chan,
            'out_chan': out_chan
        }

    def forward(self, input, sampling=None):
        y = self.convbn(input)
        y = self.resize(y)

        if sampling is None:
            # only for ini
            return y

        is_activate = (int)(sampling.item())
        if is_activate == 1:
            return y
        else:
            return torch.zeros(y.shape, device=y.device)


# 从高分辨率到低分辨率转换
class DownsampleTransformer(NetworkBlock):
    def __init__(self, in_chan, out_chan, scale):
        super(DownsampleTransformer, self).__init__()
        self.convbn =\
            ConvBn(in_chan*HRCellBlock.expansion,
                   out_chan*HRCellBlock.expansion,
                   relu=True,
                   k_size=3,
                   stride=(int)(scale))
        self.structure_fixed = False

        self.params = {
            'module_list': ['DownsampleTransformer'],
            'name_list': ['DownsampleTransformer'],
            'DownsampleTransformer': {'out_chan': out_chan,
                                    'in_chan': in_chan,
                                    'scale': scale,},
            'in_chan': in_chan,
            'out_chan': out_chan
        }

    def forward(self, input, sampling=None):
        y = self.convbn(input)

        if sampling is None:
            # only for ini
            return y

        is_activate = (int)(sampling.item())
        if is_activate == 1:
            return y
        else:
            return torch.zeros(y.shape, device=y.device)


# 聚合
class AddAggregate(NetworkBlock):
    def __init__(self):
        super(AddAggregate, self).__init__()
        self.structure_fixed = False
        self.params = {
            'module_list': ['AddAggregate'],
            'name_list': ['AddAggregate'],
            'AddAggregate': {},
            'in_chan': 0,
            'out_chan': 0
        }

    def forward(self, x, sampling=None):
        if not isinstance(x, list):
            x = [x]

        y = F.relu(sum(x))
        if sampling is None:
            return y

        is_activate = (int)(sampling.item())
        if is_activate == 1:
            return y
        else:
            return torch.zeros(y.shape, device=y.device)


class PKHRNetSN(ContinuousSuperNetwork):
    def __init__(self,
                 data_prop,
                 *args, **kwargs):
        super(PKHRNetSN, self).__init__(*args, **kwargs)
        NetworkBlock.state_num = 2
        self.in_chan = data_prop['in_channels']
        self.in_size = data_prop['img_dim']
        self.out_dim = data_prop['out_size'][0]

        self._input_size = (self.in_chan, self.in_size, self.in_size)
        self._criterion = seg_cross_entropy
        self.graph = nx.DiGraph()
        self.sampling_parameters = nn.ParameterList()

        # head (固定计算节点，对应激活参数不可学习)
        head_1 = ConvBn(self.in_chan, 64, k_size=3, stride=2, relu=True)
        head_2 = ConvBn(64, 64, k_size=3, stride=2, relu=True)

        # tail (固定计算节点，结构不可学习)
        tail = kwargs['out_layer']

        # 构建超网络结构
        self.grid_arc = \
            GridArc(self.graph,
                    grid_h=4,
                    grid_w=6,
                    cell_cls=HRCellBlock,
                    skip_transformer_cls=SameTransformer,
                    upsample_transformer_cls=UpsampleTransformer,
                    downsample_transformer_cls=DownsampleTransformer,
                    add_aggregate_cls=AddAggregate,
                    num_blocks=5,
                    num_channels=[48, 96, 192, 384])
        in_name, out_name = self.grid_arc.generate([head_1, head_2], tail)

        self.using_static_arch = False
        self.blocks = self.grid_arc.blocks
        self.set_graph(self.graph, in_name, out_name)

        # 保存搜索空间图
        a = NASDrawer()
        a.draw(self.graph, filename='./searchspace.svg')

    @property
    def criterion(self):
        return self._criterion

    @property
    def accuracy_evaluator(self):
        return SegmentationAccuracyEvaluator

    def hierarchical(self):
        return []
