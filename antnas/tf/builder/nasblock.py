import tensorflow as tf

slim = tf.contrib.slim


def hard_swish(inputs):
    result = inputs * tf.nn.relu6(inputs + 3.) / 6.
    return result


def hard_sigmoid(inputs):
    result = tf.nn.relu6(inputs + 3.) / 6.
    return result


class NasBlock(object):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


def _make_divisible(v, divisor=8, min_value=8):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidualBlockWithSEHS(NasBlock):
    def __init__(self):
        super(InvertedResidualBlockWithSEHS, self).__init__()

    def __call__(self,
                 inputs,
                 expansion,
                 kernel_size,
                 out_chan,
                 reduction,
                 ratio=4,
                 se=True,
                 hs=True,
                 scope=None, **kwargs):
        with tf.variable_scope(scope, 'irb', [inputs]):
            x = inputs
            input_shape = inputs.get_shape().as_list()
            if expansion != 1:
                expansion_channels = _make_divisible(input_shape[3] * expansion)
            
                # expand no bias
                x = slim.conv2d(x,
                                num_outputs=expansion_channels,
                                kernel_size=1,
                                stride=1,
                                activation_fn=hard_swish if hs else tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                padding='SAME')
        
            # stride = kwargs.get('stride', 1)
            # if reduction and stride == 1:
            #     stride = 2
            stride = 1
            if reduction:
                stride = 2
        
            # depthwise no bias
            rate = kwargs.get('dilation', 1)
            x = slim.separable_conv2d(x,
                                      num_outputs=None,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      rate=rate,
                                      activation_fn=hard_swish if hs else tf.nn.relu,
                                      normalizer_fn=slim.batch_norm,
                                      depth_multiplier=1,
                                      padding='SAME')
        
            if se:
                x_channels = x.get_shape().as_list()[3]
                squeeze_channels = _make_divisible(x_channels / ratio, divisor=8)
            
                se_x = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)
                se_x = slim.conv2d(se_x,
                                   num_outputs=squeeze_channels,
                                   kernel_size=1,
                                   stride=1,
                                   activation_fn=tf.nn.relu,
                                   normalizer_fn=None,
                                   padding='SAME')
                se_x = slim.conv2d(se_x,
                                   num_outputs=x_channels,
                                   kernel_size=1,
                                   stride=1,
                                   activation_fn=hard_sigmoid,
                                   normalizer_fn=None,
                                   padding='SAME')
                x = tf.multiply(se_x, x)
        
            # pointwise no bias
            x = slim.conv2d(x,
                            num_outputs=out_chan,
                            kernel_size=1,
                            stride=1,
                            activation_fn=None,
                            normalizer_fn=slim.batch_norm,
                            padding='SAME')

            skip = kwargs.get('skip', True)
            if input_shape[-1] == out_chan and (stride == 1) and skip:
                x = tf.add(x, inputs)
            return x


class ConcatBlock(NasBlock):
    def __init__(self):
        super(ConcatBlock, self).__init__()

    def __call__(self, inputs, scope, **kwargs):
        if not isinstance(inputs, list):
            with tf.variable_scope(scope, 'ConcatBlock', [inputs]):
                return inputs

        with tf.variable_scope(scope, 'ConcatBlock', inputs):
            return tf.concat(inputs, axis=3)


class SepConvBN(NasBlock):
    def __init__(self):
        super(SepConvBN, self).__init__()

    def __call__(self, inputs, out_chan, k_size, stride, relu, dilation=1, scope=None, **kwargs):
        # in tensorflow, bn and bias couldnt exist in the same time
        with tf.variable_scope(scope, 'SepConvBN', [inputs]):
            x = slim.separable_conv2d(inputs,
                                      num_outputs=None,
                                      kernel_size=k_size,
                                      stride=stride,
                                      activation_fn=tf.nn.relu,
                                      normalizer_fn=slim.batch_norm,
                                      rate=dilation,
                                      padding='SAME')
            x = slim.conv2d(x,
                            num_outputs=out_chan,
                            kernel_size=1,
                            activation_fn=tf.nn.relu if relu else None,
                            normalizer_fn=slim.batch_norm)
            return x


class Identity(NasBlock):
    def __init__(self, ):
        super(Identity, self).__init__()

    def __call__(self, inputs, out_chan, scope, **kwargs):
        with tf.variable_scope(scope, 'Identity', [inputs]):
            inputs = tf.identity(inputs)
            input_shape = inputs.get_shape().as_list()
            if out_chan is not None and int(input_shape[-1]) != out_chan:
                # 1x1 conv
                inputs = slim.conv2d(inputs,
                                     num_outputs=out_chan,
                                     kernel_size=1,
                                     stride=1,
                                     activation_fn=None,
                                     normalizer_fn=slim.batch_norm,
                                     rate=1,
                                     padding='SAME')

            return inputs


class ConvBn(NasBlock):
    def __init__(self):
        super(ConvBn, self).__init__()

    def __call__(self, inputs, out_chan, k_size, stride, relu, dilation=1, scope=None, **kwargs):
        # in tensorflow, bn and bias couldnt exist in the same time
        with tf.variable_scope(scope, 'ConvBn', [inputs]):
            return slim.conv2d(inputs,
                               num_outputs=out_chan,
                               kernel_size=k_size,
                               stride=stride,
                               activation_fn=tf.nn.relu if relu else None,
                               normalizer_fn=slim.batch_norm,
                               rate=dilation,
                               padding='SAME')


class AddBlock(NasBlock):
    def __init__(self):
        super(AddBlock, self).__init__()

    def __call__(self, inputs, scope, **kwargs):
        if not isinstance(inputs, list):
            with tf.variable_scope(scope, 'AddBlock', [inputs]):
                # inputs = tf.identity(inputs)
                return inputs

        with tf.variable_scope(scope, 'AddBlock', inputs):
            ss = {}
            dd = {}
            for i in range(len(inputs)):
                if inputs[i].name not in ss:
                    ss[inputs[i].name] = inputs[i]
                    dd[inputs[i].name] = 1
                else:
                    dd[inputs[i].name] = dd[inputs[i].name] + 1

            a = None
            for k, v in ss.items():
                if a is None:
                    a = v * dd[k]
                else:
                    a = tf.add(a, v * dd[k])

            return a


class Skip(NasBlock):
    def __init__(self):
        super(Skip, self).__init__()

    def __call__(self, inputs, out_chan, reduction, scope, **kwargs):
        with tf.variable_scope(scope, 'Skip', [inputs]):
            inputs = tf.identity(inputs)
            input_shape = inputs.get_shape()
            if out_chan > int(input_shape[-1]):
                pad = tf.zeros((int(input_shape[0]),
                                int(input_shape[1]),
                                int(input_shape[2]),
                                out_chan - int(input_shape[3])))
                inputs = tf.concat((inputs, pad), axis=3)
            elif out_chan < int(input_shape[-1]):
                inputs = inputs[:, :, :, 0:out_chan]
        
            if reduction:
                inputs = slim.avg_pool2d(inputs, kernel_size=2, stride=2)
        
            return inputs


class ResizedBlock(NasBlock):
    def __init__(self):
        super(ResizedBlock, self).__init__()

    def __call__(self, inputs, out_chan, relu, k_size, scale_factor, scope, **kwargs):
        with tf.variable_scope(scope, 'Resized', [inputs]):
            input_shape = inputs.get_shape().as_list()
            x = tf.image.resize_images(inputs,
                                       [int(input_shape[1] * scale_factor), int(input_shape[2] * scale_factor)],
                                       align_corners=True)
            if out_chan > 0:
                x = slim.conv2d(x,
                                num_outputs=out_chan,
                                kernel_size=k_size,
                                stride=1,
                                activation_fn=tf.nn.relu if relu else None,
                                normalizer_fn=slim.batch_norm,
                                padding='SAME')

            return x


class ASPPBlock(NasBlock):
    def __init__(self):
        super(ASPPBlock, self).__init__()

    def __call__(self, inputs, in_chan, depth, atrous_rates, scope, **kwargs):
        with tf.variable_scope(scope, 'ASPP', [inputs]):
            input_shape = inputs.get_shape().as_list()

            part_list = []
            # part 1
            part_1 = tf.reduce_mean(inputs, axis=[1, 2], keep_dims=True)
            part_1 = slim.conv2d(part_1,
                                 num_outputs=depth,
                                 kernel_size=1,
                                 stride=1,
                                 activation_fn=tf.nn.relu,
                                 normalizer_fn=slim.batch_norm,
                                 padding='SAME')
            part_1 = tf.image.resize_images(part_1, size=[input_shape[1], input_shape[2]], align_corners=True)
            part_list.append(part_1)

            # part 2
            part_2 = slim.conv2d(inputs,
                                 num_outputs=depth,
                                 kernel_size=1,
                                 stride=1,
                                 activation_fn=tf.nn.relu,
                                 normalizer_fn=slim.batch_norm,
                                 padding='SAME')
            part_list.append(part_2)

            # part 3 ~
            for i in range(len(atrous_rates)):
                part_i = slim.separable_conv2d(inputs,
                                               num_outputs=depth,
                                               kernel_size=3,
                                               stride=1,
                                               activation_fn=tf.nn.relu,
                                               normalizer_fn=slim.batch_norm,
                                               rate=atrous_rates[i],
                                               padding='SAME')
                part_list.append(part_i)

            # concat all parts
            part = tf.concat(part_list, axis=3)
            res = slim.conv2d(part,
                              num_outputs=depth,
                              kernel_size=1,
                              activation_fn=tf.nn.relu,
                              normalizer_fn=slim.batch_norm,
                              padding='SAME')

            return res


class LargekernelConv(NasBlock):
    def __init__(self):
        super(LargekernelConv, self).__init__()

    def __call__(self, inputs, out_chan, k_size, bias, scope):
        with tf.variable_scope(scope, 'LargekernelConv', [inputs]):
            left_x = slim.conv2d(inputs,
                                 num_outputs=out_chan,
                                 kernel_size=[k_size, 1],
                                 stride=1,
                                 activation_fn=None,
                                 normalizer_fn=None,
                                 padding='SAME')
            left_x = slim.conv2d(left_x,
                                 num_outputs=out_chan,
                                 kernel_size=[1, k_size],
                                 stride=1,
                                 activation_fn=None,
                                 normalizer_fn=None,
                                 padding='SAME')

            right_x = slim.conv2d(inputs,
                                  num_outputs=out_chan,
                                  kernel_size=[1, k_size],
                                  stride=1,
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  padding='SAME')
            right_x = slim.conv2d(right_x,
                                  num_outputs=out_chan,
                                  kernel_size=[k_size, 1],
                                  stride=1,
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  padding='SAME')

            x = left_x + right_x

            br_x = slim.conv2d(x,
                               num_outputs=out_chan,
                               kernel_size=3,
                               stride=1,
                               activation_fn=tf.nn.relu,
                               normalizer_fn=None,
                               padding='SAME')

            br_x = slim.conv2d(br_x,
                               num_outputs=out_chan,
                               kernel_size=3,
                               stride=1,
                               activation_fn=None,
                               normalizer_fn=None,
                               padding='SAME')

            return tf.add(x, br_x)

