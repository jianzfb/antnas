# -*- coding: UTF-8 -*-
# @Time    : 2020-06-08 09:03
# @File    : ENAS.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antnas.searchspace.PKAutoArc import *
from antnas.searchspace.PKArc import *
from antnas.component.NetworkBlock import *
from antnas.component.NetworkBlock import _make_divisible
from OutLayerFactory import *
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ENAS
def ENAS(tail, prefix):
    pk = PKArc()

    # sep1_r = SeparableConv2D(64, (5, 5), use_bias=False, name='sep1', padding='same', activation='relu')(inp)
    # sep1 = BatchNormalization()(sep1_r)
    sep_1 = SepConvBN(3, 64, relu=True, k_size=5)
    pk.add(sep_1, "sep_1")

    # conv1_r = Convolution2D(64, (5, 5), use_bias=False, name='conv1', padding='same', activation='relu')(sep1)
    # conv1 = BatchNormalization()(conv1_r)
    conv_1 = ConvBn(64, 64, relu=True, k_size=5)
    pk.add(conv_1, "conv_1")
    pk.link("sep_1", "conv_1")

    # conv2_r = Convolution2D(64, (5, 5), use_bias=False, name='conv2', padding='same', activation='relu')(conv1)
    # conv2 = BatchNormalization()(conv2_r)
    conv_2 = ConvBn(64, 64,relu=True,k_size=5)
    pk.add(conv_2, "conv_2")
    pk.link("conv_1", "conv_2")

    # concat1_r = merge_layers([sep1, conv2])
    # concat1 = BatchNormalization()(concat1_r)
    merge_1 = MergeBlock(128, 64)
    pk.add(merge_1, "merge_1")
    pk.link("sep_1", "merge_1")
    pk.link("conv_2", "merge_1")

    # sep2_r = SeparableConv2D(64, (5, 5), use_bias=False, name='sep2', padding='same', activation='relu')(concat1)
    # sep2 = BatchNormalization()(sep2_r)
    sep_2 = SepConvBN(64, 64, relu=True, k_size=5)
    pk.add(sep_2, "sep_2")
    pk.link("merge_1", "sep_2")

    # concat2_r = merge_layers([sep1, sep2])
    # concat2 = BatchNormalization()(concat2_r)
    merge_2 = MergeBlock(128, 64)
    pk.add(merge_2, "merge_2")
    pk.link("sep_1", "merge_2")
    pk.link("sep_2", "merge_2")

    # pool1 = MaxPooling2D(2, 2, name='pool1')(concat2)
    pool_1 = MaxPoolingBlock(2,2)
    pk.add(pool_1, "pool1")
    pk.link("merge_2", "pool1")

    # sep3_r = SeparableConv2D(128, (3, 3), use_bias=False, name='sep3', padding='same', activation='relu')(pool1)
    # sep3 = BatchNormalization()(sep3_r)
    sep_3 = SepConvBN(64, 128, relu=True, k_size=3)
    pk.add(sep_3, "sep_3")
    pk.link("pool1", "sep_3")

    # concat4_r = merge_layers([sep2, sep3])
    # concat4 = BatchNormalization()(concat4_r)
    merge_4 = MergeBlock(192, 64)
    pk.add(merge_4, "merge_4")
    pk.link("sep_2", "merge_4")
    pk.link("sep_3", "merge_4")

    # conv3_r = Convolution2D(128, (5, 5), use_bias=False, name='conv3', padding='same', activation='relu')(concat4)
    # conv3 = BatchNormalization()(conv3_r)
    conv_3 = ConvBn(64, 128, relu=True, k_size=5)
    pk.add(conv_3, "conv_3")
    pk.link("merge_4", "conv_3")

    # concat5_r = merge_layers([conv2, sep2, sep3, conv3])
    # concat5 = BatchNormalization()(concat5_r)
    # 64+64+128+128
    merge_5 = MergeBlock(384, 64)
    pk.add(merge_5, "merge_5")
    pk.link("conv_2", "merge_5")
    pk.link("sep_2", "merge_5")
    pk.link("sep_3", "merge_5")
    pk.link("conv_3", "merge_5")

    # sep4_r = SeparableConv2D(128, (3, 3), use_bias=False, name='sep4', padding='same', activation='relu')(concat5)
    # sep4 = BatchNormalization()(sep4_r)
    sep_4 = SepConvBN(64, 128, relu=True, k_size=3)
    pk.add(sep_4, "sep_4")
    pk.link("merge_5", "sep_4")

    # concat6_r = merge_layers([conv2, conv3, sep1, sep2, sep3, sep4])
    # concat6 = BatchNormalization()(concat6_r)
    # 64+128+64+64+128+128
    merge_6 = MergeBlock(576, 64)
    pk.add(merge_6, "merge_6")
    pk.link("conv_2", "merge_6")
    pk.link("conv_3", "merge_6")
    pk.link("sep_1", "merge_6")
    pk.link("sep_2", "merge_6")
    pk.link("sep_3", "merge_6")
    pk.link("sep_4", "merge_6")

    # sep5_r = SeparableConv2D(128, (5, 5), use_bias=False, name='sep5', padding='same', activation='relu')(concat6)
    # sep5 = BatchNormalization()(sep5_r)
    sep_5 = SepConvBN(64, 128, relu=True, k_size=5)
    pk.add(sep_5, "sep_5")
    pk.link("merge_6", "sep_5")

    # concat7_r = merge_layers([sep1, sep2, sep3, sep4, sep5])
    # concat7 = BatchNormalization()(concat7_r)
    # 64+64+128+128+128
    merge_7 = MergeBlock(512, 64)
    pk.add(merge_7, "merge_7")
    pk.link("sep_1", "merge_7")
    pk.link("sep_2", "merge_7")
    pk.link("sep_3", "merge_7")
    pk.link("sep_4", "merge_7")
    pk.link("sep_5", "merge_7")

    pool2 = MaxPoolingBlock(k_size=2, stride=2)
    pk.add(pool2, "pool2")
    pk.link("merge_7", "pool2")

    # concat8_r = merge_layers([conv3, pool2])
    # concat8 = BatchNormalization()(concat8_r)
    # 128+64
    merge_8 = MergeBlock(192, 64)
    pk.add(merge_8, "merge_8")
    pk.link("conv_3", "merge_8")
    pk.link("pool2", "merge_8")

    # conv4_r = Convolution2D(256, (5, 5), use_bias=False, name='conv4', padding='same', activation='relu')(concat8)
    # conv4 = BatchNormalization()(conv4_r)
    conv_4 = ConvBn(64, 256, relu=True, k_size=5)
    pk.add(conv_4, "conv_4")
    pk.link("merge_8", "conv_4")

    # concat9_r = merge_layers([sep2, sep4, conv2, conv4])
    # concat9 = BatchNormalization()(concat9_r)
    # 64+128+64+256
    merge_9 = MergeBlock(512, 64)
    pk.add(merge_9, "merge_9")
    pk.link("sep_2", "merge_9")
    pk.link("sep_4", "merge_9")
    pk.link("conv_2", "merge_9")
    pk.link("conv_4", "merge_9")

    # sep6_r = SeparableConv2D(256, (5, 5), use_bias=False, name='sep6', padding='same', activation='relu')(concat9)
    # sep6 = BatchNormalization()(sep6_r)
    sep_6 = SepConvBN(64,256,relu=True,k_size=5)
    pk.add(sep_6, "sep_6")
    pk.link("merge_9", "sep_6")

    # concat10_r = merge_layers([sep3, conv1, conv2, conv4, sep6])
    # concat10 = BatchNormalization()(concat10_r)
    # 128+64+64+256+256
    merge_10 = MergeBlock(768, 64)
    pk.add(merge_10,"merge_10")
    pk.link("sep_3", "merge_10")
    pk.link("conv_1", "merge_10")
    pk.link("conv_2", "merge_10")
    pk.link("conv_4", "merge_10")
    pk.link("sep_6", "merge_10")

    # conv5_r = Convolution2D(256, (3, 3), use_bias=False, name='conv5', padding='same', activation='relu')(concat10)
    # conv5 = BatchNormalization()(conv5_r)
    conv_5 = ConvBn(64,256, relu=True, k_size=3)
    pk.add(conv_5, "conv_5")
    pk.link("merge_10", "conv_5")

    # concat11_r = merge_layers([conv2, sep1, sep2, sep3, sep4, sep6, conv5])
    # concat11 = BatchNormalization()(concat11_r)
    # 64+64+64+128+128+256+256
    merge_11 = MergeBlock(960,64)
    pk.add(merge_11, "merge_11")
    pk.link("conv_2", "merge_11")
    pk.link("sep_1", "merge_11")
    pk.link("sep_2", "merge_11")
    pk.link("sep_3", "merge_11")
    pk.link("sep_4", "merge_11")
    pk.link("sep_6", "merge_11")
    pk.link("conv_5", "merge_11")

    # sep7_r = SeparableConv2D(256, (5, 5), use_bias=False, name='sep7', padding='same', activation='relu')(concat11)
    # sep7 = BatchNormalization()(sep7_r)
    sep_7 = SepConvBN(64,256,relu=True,k_size=5)
    pk.add(sep_7, "sep_7")
    pk.link("merge_11", "sep_7")

    # concat12_r = merge_layers([sep4, sep2, sep6, sep7])
    # concat12 = BatchNormalization()(concat12_r)
    # 128+64+256+256
    merge_12 = MergeBlock(704,64)
    pk.add(merge_12, "merge_12")
    pk.link("sep_4", "merge_12")
    pk.link("sep_2", "merge_12")
    pk.link("sep_6", "merge_12")
    pk.link("sep_7", "merge_12")

    pk.generate(None, tail)
    pk.save('./', '%s'%prefix)
    return pk


# # cifar ENAS
# # tail (固定计算节点，结构不可学习)
# tail = OutLayer(64, (10,), True)
# pk = ENAS(tail, "ENAS")
# sampled_loss, pruned_loss = pk.arc_loss([1,3,32,32], 'comp')
# print(pruned_loss)
