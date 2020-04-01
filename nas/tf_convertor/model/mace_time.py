# @Time    : 2019/10/22 10:43
# @Author  : zhangchenming
from nas.tf_convertor.builder.nasfactory import *


def gettime(images, is_training, arch):
    with tf.contrib.slim.arg_scope(training_scope(is_training=is_training)):
        params_index = int(arch)
        params = [{'hs': True, 'se': True, 'expansion': 6, 'ksize': 5},
                  {'hs': True, 'se': True, 'expansion': 6, 'ksize': 3},
                  {'hs': True, 'se': True, 'expansion': 3, 'ksize': 5},
                  {'hs': True, 'se': True, 'expansion': 3, 'ksize': 3},

                  {'hs': True, 'se': False, 'expansion': 6, 'ksize': 5},
                  {'hs': True, 'se': False, 'expansion': 6, 'ksize': 3},
                  {'hs': True, 'se': False, 'expansion': 3, 'ksize': 5},
                  {'hs': True, 'se': False, 'expansion': 3, 'ksize': 3},

                  {'hs': False, 'se': True, 'expansion': 6, 'ksize': 5},
                  {'hs': False, 'se': True, 'expansion': 6, 'ksize': 3},
                  {'hs': False, 'se': True, 'expansion': 3, 'ksize': 5},
                  {'hs': False, 'se': True, 'expansion': 3, 'ksize': 3},

                  {'hs': False, 'se': False, 'expansion': 6, 'ksize': 5},
                  {'hs': False, 'se': False, 'expansion': 6, 'ksize': 3},
                  {'hs': False, 'se': False, 'expansion': 3, 'ksize': 5},
                  {'hs': False, 'se': False, 'expansion': 3, 'ksize': 3}
                  ]

        blocks_per_stage = [1, 1, 2, 2]
        cells_per_block = [[2], [2], [2, 2], [2, 2]]
        channels_per_block = [[16], [32], [64, 96], [112, 160]]

        feature = ConvBn()(images, 16, 3, 2, relu=True)
        for stage_i in range(len(blocks_per_stage)):
            if stage_i == len(blocks_per_stage) - 1:
                is_last_stage = True
            else:
                is_last_stage = False
            feature = add_stage(feature, blocks_per_stage[stage_i], cells_per_block[stage_i], channels_per_block[stage_i], is_last_stage, params[params_index])

        feature = slim.conv2d(feature, num_outputs=feature.get_shape()[-1], kernel_size=7)
        feature = ConvBn()(feature, out_chan=21, k_size=1, stride=1, relu=True)
        feature = tf.image.resize_bilinear(feature, [512, 512], align_corners=True)

        return feature


def add_stage(inputs, block_num, cells_per_block, channels_per_block, is_last_stage, param):
    feature = inputs
    for block_i in range(block_num):
        if block_i == 0:
            is_first_block = True
        else:
            is_first_block = False
        feature = add_block(feature, cells_per_block[block_i], channels_per_block[block_i], is_first_block, is_last_stage, param)

    return feature


def add_block(inputs, cells, channles, is_first_block=False, is_last_stage=False, param=None):
    if param is None:
        param = {}
    feature = inputs

    for cell_i in range(cells):
        if cell_i == 0 and is_first_block and not is_last_stage:
            reduction = True
        else:
            reduction = False
        feature = add_cell(feature, channles, reduction, param)

    return feature


def add_cell(inputs, out_chan, reduction, param=None):
    if param is None:
        param = {}
    c = inputs.get_shape()[-1]
    feature = slim.conv2d(inputs, num_outputs=c, kernel_size=7)
    feature = InvertedResidualBlockWithSEHS()(inputs=feature,
                                              expansion=param['expansion'],
                                              kernel_size=param['ksize'],
                                              out_chan=out_chan,
                                              reduction=reduction,
                                              se=param['se'],
                                              hs=param['hs'],
                                              scope=None)

    print(feature.get_shape().as_list())

    return feature


if __name__ == '__main__':
    input_image = tf.random_uniform([1, 512, 512, 3])
    gettime(input_image, False, arch="")
