# -*- coding: UTF-8 -*-
# @Time    : 2019-09-20 20:01
# @File    : advancednasblock.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from nas.tools.nasblock import *
import tensorflow as tf
slim = tf.contrib.slim


class ASPPBlock(NasBlock):
    def __init__(self):
        super(ASPPBlock, self).__init__()

    def __call__(self, inputs, in_chan, depth, atrous_rates, scope):
        with tf.variable_scope('ASPP_%s' % scope, 'ASPP', [inputs]):
            input_shape = inputs.get_shape().as_list()

            part_list = []
            # part 1
            part_1 = tf.reduce_mean(inputs, axis=[1, 2], keep_dims=True)
            part_1 = slim.conv2d(part_1,
                                 num_outputs=depth,
                                 kernel_size=1,
                                 stride=1,
                                 activation_fn=tf.nn.relu6,
                                 normalizer_fn=slim.batch_norm,
                                 padding='SAME')
            part_1 = tf.image.resize_images(part_1, size=[input_shape[1], input_shape[2]], align_corners=True)
            part_list.append(part_1)

            # part 2
            part_2 = slim.conv2d(inputs,
                                 num_outputs=depth,
                                 kernel_size=1,
                                 stride=1,
                                 activation_fn=tf.nn.relu6,
                                 normalizer_fn=slim.batch_norm,
                                 padding='SAME')
            part_list.append(part_2)

            # part 3 ~
            for i in range(len(atrous_rates)):
                part_i = slim.separable_conv2d(inputs,
                                               num_outputs=depth,
                                               kernel_size=3,
                                               stride=1,
                                               activation_fn=tf.nn.relu6,
                                               normalizer_fn=slim.batch_norm,
                                               rate=atrous_rates[i],
                                               padding='SAME')
                part_list.append(part_i)

            # concat all parts
            part = tf.concat(part_list, 3)
            res = slim.conv2d(part,
                              num_outputs=depth,
                              kernel_size=1,
                              activation_fn=tf.nn.relu6,
                              normalizer_fn=slim.batch_norm,
                              padding='SAME')
            return res


class FocusBlock(NasBlock):
    def __init__(self):
        super(FocusBlock, self).__init__()

    def __call__(self, inputs, in_chan, out_chan, scope):
        with tf.variable_scope('Focus_%s' % scope, 'Focus', [inputs]):
            sep_conv1 = slim.separable_conv2d(inputs,
                                              num_outputs=out_chan,
                                              kernel_size=3,
                                              stride=1,
                                              activation_fn=tf.nn.relu6,
                                              normalizer_fn=slim.batch_norm,
                                              rate=1,
                                              padding='SAME')

            sep_conv2 = slim.separable_conv2d(sep_conv1,
                                              num_outputs=out_chan,
                                              kernel_size=3,
                                              stride=1,
                                              activation_fn=tf.nn.relu6,
                                              normalizer_fn=slim.batch_norm,
                                              rate=2,
                                              padding='SAME')

            sep_conv3 = slim.separable_conv2d(sep_conv2,
                                              num_outputs=out_chan,
                                              kernel_size=3,
                                              stride=1,
                                              activation_fn=tf.nn.relu6,
                                              normalizer_fn=slim.batch_norm,
                                              rate=4,
                                              padding='SAME')
            return sep_conv1 + sep_conv2 + sep_conv3


class GCN(NasBlock):
    def __init__(self):
        super(GCN, self).__init__()

    def __call__(self, inputs, out_chan, k_size, bias, scope):
        with tf.variable_scope('GCN_%s' % scope, 'GCN', [inputs]):
            left_conv1 = slim.conv2d(inputs,
                                 num_outputs=out_chan,
                                 kernel_size=[k_size, 1],
                                 stride=1,
                                 activation_fn=None,
                                 normalizer_fn=None,
                                 padding='SAME')
            left_conv2 = slim.conv2d(left_conv1,
                                 num_outputs=out_chan,
                                 kernel_size=[1, k_size],
                                 stride=1,
                                 activation_fn=None,
                                 normalizer_fn=None,
                                 padding='SAME')

            right_conv1 = slim.conv2d(inputs,
                                 num_outputs=out_chan,
                                 kernel_size=[1, k_size],
                                 stride=1,
                                 activation_fn=None,
                                 normalizer_fn=None,
                                 padding='SAME')
            right_conv2 = slim.conv2d(right_conv1,
                                 num_outputs=out_chan,
                                 kernel_size=[k_size, 1],
                                 stride=1,
                                 activation_fn=None,
                                 normalizer_fn=None,
                                 padding='SAME')

            x = left_conv2 + right_conv2

            x_res = slim.conv2d(x,
                                 num_outputs=out_chan,
                                 kernel_size=[3,3],
                                 stride=1,
                                 activation_fn=tf.nn.relu6,
                                 normalizer_fn=None,
                                 padding='SAME')
            x_res = slim.conv2d(x_res,
                                num_outputs=out_chan,
                                kernel_size=[3, 3],
                                stride=1,
                                activation_fn=None,
                                normalizer_fn=None,
                                padding='SAME'
                                )

            x_res = x_res + x
            return x_res


class BoundaryRefinement(NasBlock):
    def __init__(self):
        super(BoundaryRefinement, self).__init__()

    def __call__(self, inputs, out_chan, scope):
        with tf.variable_scope('BoundaryRefinement_%s' % scope, 'BoundaryRefinement', [inputs]):
            x_res = slim.conv2d(inputs,
                                 num_outputs=out_chan,
                                 kernel_size=[3,3],
                                 stride=1,
                                 activation_fn=tf.nn.relu6,
                                 normalizer_fn=None,
                                 padding='SAME')
            x_res = slim.conv2d(x_res,
                                num_outputs=out_chan,
                                kernel_size=[3, 3],
                                stride=1,
                                activation_fn=None,
                                normalizer_fn=None,
                                padding='SAME'
                                )

            x_res = x_res + inputs
            return x_res


