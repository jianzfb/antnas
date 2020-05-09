# -*- coding: UTF-8 -*-
# @Time    : 2020-04-07 16:32
# @File    : anchorarchs.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from nas.networks.Anchors import *
from nas.searchspace.StageBlockCellArc import *
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OutLayer(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, in_chan, out_shape, bias=True):
        super(OutLayer, self).__init__()
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.conv_1 = nn.Conv2d(in_chan, 960, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn = nn.BatchNorm2d(960)

        self.conv_2 = nn.Conv2d(960, 1280, kernel_size=1, stride=1, padding=0, bias=bias)
        self.conv_3 = nn.Conv2d(1280, out_shape[0], kernel_size=1, stride=1, padding=0, bias=bias)
        self.out_shape = out_shape
        self.params = {
            'module_list': ['OutLayer'],
            'name_list': ['OutLayer'],
            'OutLayer': {'out_shape': out_shape, 'in_chan': in_chan},
            'out': 'outname'
        }

    def forward(self, x, sampling=None):
        x = self.conv_1(x)
        x = self.bn(x)
        x = F.relu6(x)

        x = self.global_pool(x)
        x = self.conv_2(x)
        x = F.relu6(x)

        x = self.conv_3(x)
        return x.view(-1, *self.out_shape)

    def get_flop_cost(self, x):
        return [0] + [0] * (self.state_num - 1)


anchornetwork_manager = AnchorNetwork()

graph = nx.DiGraph()
sbca = StageBlockCellArc(CellBlock, ReductionCellBlock, AddBlock, ConvBn, graph)

# head (固定计算节点，对应激活参数不可学习)
head = ConvBn(3, 32, k_size=3, stride=1, relu=True)
# tail (固定计算节点，结构不可学习)
tail = OutLayer(256, (10,), True)

anchornetwork_manager.generate(arch_generator=sbca,
                               anchor_num=5,
                               folder="./supernetwork/",
                               head=head,
                               tail=tail,
                               blocks=[2, 2, 1],
                               cells=[[4, 4], [4, 4], [3]],
                               channels=[[32, 32], [64, 128], [256]])



