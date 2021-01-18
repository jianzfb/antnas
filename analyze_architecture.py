# -*- coding: UTF-8 -*-
# @Time    : 2020/12/2 4:41 下午
# @File    : analyze_architecture.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antnas.searchspace.LoadArc import *
from sota.mobilenet_v3 import *
from antnas.utils.misc import *
from antnas.networks.FixedNetwork import *
import argparse
from thop import profile


parser = argparse.ArgumentParser()
parser.add_argument('--architecture', type=str, default='')
parser.add_argument('--cost_evaluation', default=['latency'], type=restricted_list('comp', 'latency', 'param'))
parser.add_argument('--devices', default='0', type=str)
parser.add_argument('--shape', default='1,3,224,224', type=str)
parser.add_argument('--latency_lookup_table', default='', type=str)


class ImageNetOutLayer(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, in_chan):
        super(ImageNetOutLayer, self).__init__()
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        # self.conv_1 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn_1 = nn.BatchNorm2d(960,
        #                            momentum=1.0 if not NetworkBlock.bn_moving_momentum else 0.1,
        #                            track_running_stats=NetworkBlock.bn_track_running_stats)
        #
        # self.conv_2 = nn.Conv2d(960, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        # self.classifier = nn.Linear(1280, 1000)

        # self.conv_1 = nn.Conv2d(160, 576, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn_1 = nn.BatchNorm2d(576,
        #                            momentum=1.0 if not NetworkBlock.bn_moving_momentum else 0.1,
        #                            track_running_stats=NetworkBlock.bn_track_running_stats)
        #
        # self.conv_2 = nn.Conv2d(576, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        # self.classifier = nn.Linear(1024, 1000)
        #
        # self.dropout = torch.nn.Dropout(p=0.9)
        #
        # self.params = {
        #     'module_list': ['ImageNetOutLayer'],
        #     'name_list': ['ImageNetOutLayer'],
        #     'ImageNetOutLayer': {'in_chan': in_chan},
        #     'out': 'outname',
        #     'in_chan': in_chan,
        # }
        pass

    def forward(self, x, sampling=None):
        # x = torch.zeros((1,160,7,7))
        # x = self.conv_1(x)
        # x = self.bn_1(x)
        # x = F.relu(x)
        #
        # x = self.global_pool(x)
        # x = self.conv_2(x)
        # x = F.relu(x)
        # x = self.dropout(x)
        #
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x


if __name__ == '__main__':
    args = parser.parse_args()

    # model path
    architecture_path = args.architecture
    cost_evaluation = args.cost_evaluation
    devices = args.devices
    shape = args.shape
    latency_lookup_table = args.latency_lookup_table

    end_pos = architecture_path.split('/')[-1].find('.architecture')
    architecture_name = architecture_path.split('/')[-1][:end_pos]

    # # 加载模型
    # pk = LoadArc(architecture_path)
    # pk.generate(tail=ImageNetOutLayer)
    # # 计算代价
    # input_shape = [(int)(s) for s in shape.split(',')]
    # for ce in cost_evaluation:
    #     sampled_loss, pruned_loss = \
    #         pk.arc_loss(input_shape,
    #                     ce,
    #                     latency_lookup_table=latency_lookup_table,
    #                     devices=[] if ce != 'latency' else [(int)(m) for m in devices.split(',')])
    #
    #     print('%s - %f'%(ce, pruned_loss[0].item()))

    model = FixedNetwork(architecture=architecture_path,
                         output_layer_cls=ImageNetOutLayer,
                         accuracy_evaluator_cls=None,
                         network_name=architecture_name)
    input_size = (1, 3, 224, 224)
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input, None))
    print('Total params: %.2fM' % (params/1000000.0))
    print('Total flops: %.2fM' % (flops/1000000.0))
