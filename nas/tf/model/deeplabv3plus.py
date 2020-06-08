import tensorflow as tf
from nas.tf.utility import train_utils
from nas.tf.config import config_param
from nas.tf.utility import get_dataset_colormap

from nas.tf.builder.nasfactory import *

slim = tf.contrib.slim


class SegOutLayer(NasBlock):
    def __init__(self):
        super(SegOutLayer, self).__init__()

    def __call__(self, inputs, out_shape, scope, **kwargs):
        with tf.variable_scope('out_%s' % scope, 'out', [inputs]):
            # x = slim.conv2d(inputs,
            #                 num_outputs=21,
            #                 kernel_size=3,
            #                 padding='SAME',
            #                 activation_fn=tf.nn.relu6,
            #                 normalizer_fn=None)
            # x = slim.conv2d(x,
            #                 num_outputs=21,
            #                 kernel_size=3,
            #                 padding='SAME',
            #                 activation_fn=None,
            #                 normalizer_fn=None)
            y = slim.conv2d(inputs,
                            num_outputs=21,
                            kernel_size=1,
                            padding='SAME',
                            activation_fn=None,
                            normalizer_fn=None)

            return y


def get_logits(images,
               weight_decay=0.0001,
               reuse=None,
               is_training=False,
               arch=''):

    with tf.contrib.slim.arg_scope(training_scope(is_training=is_training)):
        nas_factory = NasFactory()
        output = nas_factory.build(images,
                                   {'outname': SegOutLayer},
                                   arch)
    logits = output['outname']
    logits = tf.image.resize_bilinear(logits, images.get_shape()[1:3], align_corners=True)
    # logits = gettime(images, is_training, arch)
    return logits


def deeplabv3_plus_model_fn(features, labels, mode, params):
    if isinstance(features, dict):
        features = features['feature']

    preprocessed_inputs = (2.0 / 255.0) * tf.cast(features, tf.float32) - 1.0
    print(preprocessed_inputs.get_shape())

    logits = get_logits(
        preprocessed_inputs,
        config_param.weight_decay,
        is_training=(mode == tf.estimator.ModeKeys.TRAIN),
        arch=params['arch']
    )

    pred_classes = tf.math.argmax(logits, axis=-1, output_type=tf.int32)
    cross_entropy = train_utils.get_cross_entry_loss(logits, labels, config_param.num_classes, config_param.ignore_label)
    l2_loss = tf.losses.get_regularization_loss()
    loss = cross_entropy + l2_loss

    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)
    tf.summary.scalar('l2_loss', l2_loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        images = tf.cast(features, tf.uint8)
        colored_label = tf.py_func(
            get_dataset_colormap.label_to_color_image, [tf.squeeze(labels)], tf.uint8)
        colored_pred = tf.py_func(
            get_dataset_colormap.label_to_color_image, [pred_classes], tf.uint8)
        tf.summary.image('images', tf.concat(axis=2, values=[images, colored_label, colored_pred]))

        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.polynomial_decay(
            config_param.base_learning_rate, global_step, config_param.training_number_of_steps,
            end_learning_rate=1e-6, power=config_param.learning_power)

        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=config_param.momentum)
        optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    pred_classes = tf.reshape(pred_classes, [-1, ])
    labels = tf.reshape(labels, [-1, ])
    weights = tf.cast(tf.not_equal(labels, config_param.ignore_label), tf.float32)
    labels = tf.where(tf.equal(labels, config_param.ignore_label), tf.zeros_like(labels), labels)

    accuracy = tf.metrics.accuracy(labels, pred_classes, weights=weights)
    tf.identity(accuracy[1], name='train_px_accuracy')
    mean_iou = tf.metrics.mean_iou(labels, pred_classes, config_param.num_classes, weights=weights)
    metrics = {'px_accuracy': accuracy, 'mean_iou': mean_iou}

    # scaffold = tf.train.Scaffold(init_fn=train_utils.get_init_fn_for_scaffold())
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics,
        scaffold=None)


if __name__ == '__main__':
    inputs_1 = tf.random_uniform((1, 512, 512, 3))
    end_points_1 = get_logits(inputs_1,
                              is_training=True,
                              arch='/Users/chmzhang/Downloads/accuray_0.8022_flops_6608158_params_432.architecture')
    print('end')

