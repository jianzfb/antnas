# @Time    : 2020/5/28 10:16
# @Author  : zhangchenming
from antnas.searchspace.PKAutoArc import *


def mobilenetv3_large(head, tail, prefix):
    modules = [
        InvertedResidualBlockWithSEHS(
            in_chan=head.params['out_chan'], out_chan=16, kernel_size=3, reduction=False, hs=False, se=False,
            expansion=1),
        InvertedResidualBlockWithSEHS(
            in_chan=16, out_chan=24, kernel_size=3, reduction=True, hs=False, se=False, expansion=4),
        InvertedResidualBlockWithSEHS(
            in_chan=24, out_chan=24, kernel_size=3, reduction=False, hs=False, se=False, expansion=3),
        InvertedResidualBlockWithSEHS(
            in_chan=24, out_chan=40, kernel_size=5, reduction=True, hs=False, se=True, expansion=3),
        InvertedResidualBlockWithSEHS(
            in_chan=40, out_chan=40, kernel_size=5, reduction=False, hs=False, se=True, expansion=3),
        InvertedResidualBlockWithSEHS(
            in_chan=40, out_chan=40, kernel_size=5, reduction=False, hs=False, se=True, expansion=3),
        InvertedResidualBlockWithSEHS(
            in_chan=40, out_chan=80, kernel_size=3, reduction=True, hs=True, se=False, expansion=6),
        InvertedResidualBlockWithSEHS(
            in_chan=80, out_chan=80, kernel_size=3, reduction=False, hs=True, se=False, expansion=2.5),
        InvertedResidualBlockWithSEHS(
            in_chan=80, out_chan=80, kernel_size=3, reduction=False, hs=True, se=False, expansion=184 / 80),
        InvertedResidualBlockWithSEHS(
            in_chan=80, out_chan=80, kernel_size=3, reduction=False, hs=True, se=False, expansion=184 / 80),
        InvertedResidualBlockWithSEHS(
            in_chan=80, out_chan=112, kernel_size=3, reduction=False, hs=True, se=True, expansion=6),
        InvertedResidualBlockWithSEHS(
            in_chan=112, out_chan=112, kernel_size=3, reduction=False, hs=True, se=True, expansion=6),
        InvertedResidualBlockWithSEHS(
            in_chan=112, out_chan=160, kernel_size=5, reduction=True, hs=True, se=True, expansion=6),
        InvertedResidualBlockWithSEHS(
            in_chan=160, out_chan=160, kernel_size=5, reduction=False, hs=True, se=True, expansion=6),
        InvertedResidualBlockWithSEHS(
            in_chan=160, out_chan=160, kernel_size=5, reduction=False, hs=True, se=True, expansion=6),
    ]

    graph = nx.DiGraph()
    pk = PKAutoArc(graph)
    pk.generate(head, tail, modules)
    pk.save('./', '%s_mobilenetv3_large' % prefix)


def mobilenetv3_small(head, tail, prefix):
    modules = [
        InvertedResidualBlockWithSEHS(
            in_chan=head.params['out_chan'], out_chan=16, kernel_size=3, reduction=True, hs=False, se=True,
            expansion=1),
        InvertedResidualBlockWithSEHS(
            in_chan=16, out_chan=24, kernel_size=3, reduction=True, hs=False, se=False, expansion=72 / 16),
        InvertedResidualBlockWithSEHS(
            in_chan=24, out_chan=24, kernel_size=3, reduction=False, hs=False, se=False, expansion=88 / 24),
        InvertedResidualBlockWithSEHS(
            in_chan=24, out_chan=40, kernel_size=5, reduction=True, hs=True, se=True, expansion=96 / 24),
        InvertedResidualBlockWithSEHS(
            in_chan=40, out_chan=40, kernel_size=5, reduction=False, hs=True, se=True, expansion=240 / 40),
        InvertedResidualBlockWithSEHS(
            in_chan=40, out_chan=40, kernel_size=5, reduction=False, hs=True, se=True, expansion=240 / 40),
        InvertedResidualBlockWithSEHS(
            in_chan=40, out_chan=48, kernel_size=5, reduction=False, hs=True, se=True, expansion=120 / 40),
        InvertedResidualBlockWithSEHS(
            in_chan=48, out_chan=48, kernel_size=5, reduction=False, hs=True, se=True, expansion=144 / 48),
        InvertedResidualBlockWithSEHS(
            in_chan=48, out_chan=96, kernel_size=5, reduction=True, hs=True, se=True, expansion=288 / 48),
        InvertedResidualBlockWithSEHS(
            in_chan=96, out_chan=96, kernel_size=5, reduction=False, hs=True, se=True, expansion=576 / 96),
        InvertedResidualBlockWithSEHS(
            in_chan=96, out_chan=96, kernel_size=5, reduction=False, hs=True, se=True, expansion=576 / 96)
    ]

    graph = nx.DiGraph()
    pk = PKAutoArc(graph)
    pk.generate(head, tail, modules)
    pk.save('./', '%s_mobilenetv3_small' % prefix)


def mobilenetv3_edgetpu(head, tail, prefix):
    modules = [
        Fused(in_chan=head.params['out_chan'], out_chan=16, expand_factor=1, relu=True, k_size=3, stride=1, dilation=1),
        Fused(in_chan=16, out_chan=32, expand_factor=8, relu=True, k_size=3, stride=2, dilation=1),
        Fused(in_chan=32, out_chan=32, expand_factor=4, relu=True, k_size=3, stride=1, dilation=1),
        Fused(in_chan=32, out_chan=32, expand_factor=4, relu=True, k_size=3, stride=1, dilation=1),
        Fused(in_chan=32, out_chan=32, expand_factor=4, relu=True, k_size=3, stride=1, dilation=1),

        Fused(in_chan=32, out_chan=48, expand_factor=8, relu=True, k_size=3, stride=2, dilation=1),
        Fused(in_chan=48, out_chan=48, expand_factor=4, relu=True, k_size=3, stride=1, dilation=1),
        Fused(in_chan=48, out_chan=48, expand_factor=4, relu=True, k_size=3, stride=1, dilation=1),
        Fused(in_chan=48, out_chan=48, expand_factor=4, relu=True, k_size=3, stride=1, dilation=1),

        InvertedResidualBlockWithSEHS(
            in_chan=48, out_chan=96, kernel_size=3, reduction=True, hs=False, se=False, expansion=8, skip=True),
        InvertedResidualBlockWithSEHS(
            in_chan=96, out_chan=96, kernel_size=3, reduction=False, hs=False, se=False, expansion=4, skip=True),
        InvertedResidualBlockWithSEHS(
            in_chan=96, out_chan=96, kernel_size=3, reduction=False, hs=False, se=False, expansion=4, skip=True),
        InvertedResidualBlockWithSEHS(
            in_chan=96, out_chan=96, kernel_size=3, reduction=False, hs=False, se=False, expansion=4, skip=True),

        InvertedResidualBlockWithSEHS(
            in_chan=96, out_chan=96, kernel_size=3, reduction=True, hs=False, se=False, expansion=8, skip=False),
        InvertedResidualBlockWithSEHS(
            in_chan=96, out_chan=96, kernel_size=3, reduction=False, hs=False, se=False, expansion=4, skip=True),
        InvertedResidualBlockWithSEHS(
            in_chan=96, out_chan=96, kernel_size=3, reduction=False, hs=False, se=False, expansion=4, skip=True),
        InvertedResidualBlockWithSEHS(
            in_chan=96, out_chan=96, kernel_size=3, reduction=False, hs=False, se=False, expansion=4, skip=True),

        InvertedResidualBlockWithSEHS(
            in_chan=96, out_chan=160, kernel_size=5, reduction=False, hs=False, se=False, expansion=8, skip=True),
        InvertedResidualBlockWithSEHS(
            in_chan=160, out_chan=160, kernel_size=3, reduction=False, hs=False, se=False, dilation=2, expansion=4,
            skip=True),
        InvertedResidualBlockWithSEHS(
            in_chan=160, out_chan=160, kernel_size=3, reduction=False, hs=False, se=False, dilation=2, expansion=4,
            skip=True),
        InvertedResidualBlockWithSEHS(
            in_chan=160, out_chan=160, kernel_size=3, reduction=False, hs=False, se=False, dilation=2, expansion=4,
            skip=True),

        InvertedResidualBlockWithSEHS(
            in_chan=160, out_chan=192, kernel_size=3, reduction=False, hs=False, se=False, dilation=2, expansion=8,
            skip=True),
    ]
