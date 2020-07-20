# -*- coding: UTF-8 -*-
# @Time    : 2020-04-07 16:32
# @File    : anchorarchs.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antnas.networks.Anchors import *
from antnas.searchspace.StageBlockCellArc import *
from antnas.searchspace.PKCifar10SN import *
from antnas.searchspace.PKCifar10SN import Cifar10CellBlock
from antnas.searchspace.PKCifar10SN import Cifar10ReductionCellBlock
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OutLayer(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, out_shape, in_chan=160, bias=True):
        super(OutLayer, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_shape[0], kernel_size=1, stride=1, padding=0, bias=bias)
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.out_shape = out_shape
        self.params = {
            'module_list': ['OutLayer'],
            'name_list': ['OutLayer'],
            'OutLayer': {'out_shape': out_shape, 'in_chan': in_chan},
            'out': 'outname'
        }

    def forward(self, x, sampling=None):
        x = self.conv(x)
        x = self.global_pool(x)
        return x.view(-1, *self.out_shape)

    def get_flop_cost(self, x):
        return [0] + [0] * (self.state_num - 1)


anchornetwork_manager = Anchors()

graph = nx.DiGraph()
sbca = StageBlockCellArc(Cifar10CellBlock,
                              Cifar10ReductionCellBlock,
                              AddBlock,
                              ConvBn,
                              graph,
                              is_cell_dense=True,
                              is_block_dense=True)
# head (固定计算节点，对应激活参数不可学习)
head = ConvBn(3, 64, k_size=3, stride=1, relu=True)
# tail (固定计算节点，结构不可学习)
tail = OutLayer((10,), 256, True)

anchornetwork_manager.generate(arch_generator=sbca,
                               input_shape=[1,3,32,32],
                               constraint=[0.8],
                               arc_loss='param',
                               folder="./supernetwork/",
                               head=head,
                               tail=tail,
                               blocks=[1, 1, 1],
                               cells=[[6], [6], [4]],
                               channels=[[64], [128], [256]])



