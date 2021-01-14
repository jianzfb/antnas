# -*- coding: UTF-8 -*-
# @Time    : 2020/12/4 1:53 下午
# @File    : test_flops.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from torchvision.models import resnet50
from thop import profile
from antnas.component.NetworkBlock import *


# 测试1：convbn flops,params
cn_1 = ConvBn(in_chan=32, out_chan=32, relu=True, k_size=3, stride=1, dilation=1)
input = torch.randn(1, 32, 224, 224)
macs, params = profile(cn_1, inputs=(input, ))
print("test - 1")
print("thop: macs %f params %f"%(macs, params))
print("antnas: macs %f params %f"%(cn_1.get_flop_cost(input)[1], cn_1.get_param_num(input)[1]))

# 测试2：convbn flops,params
cn_2 = ConvBn(in_chan=32, out_chan=64, relu=True, k_size=3, stride=2, dilation=1)
input = torch.randn(1, 32, 224, 224)
macs, params = profile(cn_2, inputs=(input, ))
print("test - 2")
print("thop: macs %f params %f"%(macs, params))
print("antnas: macs %f params %f"%(cn_2.get_flop_cost(input)[1], cn_2.get_param_num(input)[1]))


# 测试3：SepConvBN flops,params
cn_3 = SepConvBN(in_chan=32, out_chan=64, relu=True, k_size=3, stride=2, dilation=1)
input = torch.randn(1, 32, 224, 224)
macs, params = profile(cn_3, inputs=(input, ))
print("test - 3")
print("thop: macs %f params %f"%(macs, params))
print("antnas: macs %f params %f"%(cn_3.get_flop_cost(input)[1], cn_3.get_param_num(input)[1]))

# 测试 4： ResizedBlock flops, params
cn_4 = ResizedBlock(in_chan=32, out_chan=64, relu=True, k_size=3, scale_factor=2)
input = torch.randn(1, 32, 224, 224)
macs, params = profile(cn_4, inputs=(input, ))
print("test - 4")
print("thop: macs %f params %f"%(macs, params))
print("antnas: macs %f params %f"%(cn_4.get_flop_cost(input)[1], cn_4.get_param_num(input)[1]))


# 测试 5： InvertedResidualBlockWithSEHS flops, params
cn_5 = \
    InvertedResidualBlockWithSEHS(in_chan=16,
                                  expansion=6,
                                  kernel_size=3,
                                  out_chan=32,
                                  skip=True,
                                  reduction=False,
                                  ratio=4,
                                  se=False,
                                  hs=False,
                                  dilation=1)
input = torch.randn(1, 16, 224, 224)
macs, params = profile(cn_5, inputs=(input, ))
print("test - 5")
print("thop: macs %f params %f"%(macs, params))
print("antnas: macs %f params %f"%(cn_5.get_flop_cost(input)[1], cn_5.get_param_num(input)[1]))


# 测试 6： InvertedResidualBlockWithSEHS flops, params
cn_6 = \
    InvertedResidualBlockWithSEHS(in_chan=32,
                                  expansion=3,
                                  kernel_size=3,
                                  out_chan=64,
                                  skip=True,
                                  reduction=False,
                                  ratio=4,
                                  se=True,
                                  hs=False,
                                  dilation=1)
input = torch.randn(1, 32, 224, 224)
macs, params = profile(cn_6, inputs=(input, ))
print("test - 6")
print("thop: macs %f params %f"%(macs, params))
print("antnas: macs %f params %f"%(cn_6.get_flop_cost(input)[1], cn_6.get_param_num(input)[1]))


# 测试 7： InvertedResidualBlockWithSEHS flops, params
cn_7 = \
    Fused(in_chan=32, out_chan=64, expand_factor=3, relu=True, k_size=3, stride=2, dilation=1)
input = torch.randn(1, 32, 224, 224)
macs, params = profile(cn_7, inputs=(input, ))
print("test - 7")
print("thop: macs %f params %f"%(macs, params))
print("antnas: macs %f params %f"%(cn_7.get_flop_cost(input)[1], cn_7.get_param_num(input)[1]))


# 测试 8: GhostBottleneck flops, params
cn_8 = GhostBottleneck(32, 64, expansion=3, stride=2, se_ratio=1.0)
input = torch.randn(1, 32, 224, 224)
macs, params = profile(cn_8, inputs=(input, ))

print("test - 8")
print("thop: macs %f params %f"%(macs, params))
print("antnas: macs %f params %f"%(cn_8.get_flop_cost(input)[1], cn_8.get_param_num(input)[1]))

# 测试9： BottleneckBlock flops, params
cn_9 = BottleneckBlock(32, 64, k_size=3, stride=2)
input = torch.randn(1, 32, 224, 224)
macs, params = profile(cn_9, inputs=(input, ))

print("test - 9")
print("thop: macs %f params %f"%(macs, params))
print("antnas: macs %f params %f"%(cn_9.get_flop_cost(input)[1], cn_9.get_param_num(input)[1]))
