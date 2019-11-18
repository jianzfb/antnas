# -*- coding: UTF-8 -*-
# @Time    : 2019-09-14 08:46
# @File    : nasfactory.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import networkx as nx
from nas.tools.advancednasblock import *
import tensorflow as tf

slim = tf.contrib.slim


class NoOpScope(object):
    """No-op context manager."""

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def safe_arg_scope(funcs, **kwargs):
    """Returns `slim.arg_scope` with all None arguments removed.

    Arguments:
      funcs: Functions to pass to `arg_scope`.
      **kwargs: Arguments to pass to `arg_scope`.

    Returns:
      arg_scope or No-op context manager.

    Note: can be useful if None value should be interpreted as "do not overwrite
      this parameter value".
    """
    filtered_args = {name: value for name, value in kwargs.items()
                     if value is not None}
    if filtered_args:
        return slim.arg_scope(funcs, **filtered_args)
    else:
        return NoOpScope()


def training_scope(is_training=True,
                   weight_decay=0.00004,
                   stddev=0.09,
                   dropout_keep_prob=0.8,
                   bn_decay=0.997):
    """Defines NAS training scope.

    Usage:
       with tf.contrib.slim.arg_scope(nasfactory.training_scope()):
         logits, endpoints = nas_factory.build(batch_data, './nas_0.architecture')

       # the network created will be trainble with dropout/batch norm
       # initialized appropriately.
    Args:
      is_training: if set to False this will ensure that all customizations are
        set to non-training mode. This might be helpful for code that is reused
        across both training/evaluation, but most of the time training_scope with
        value False is not needed. If this is set to None, the parameters is not
        added to the batch_norm arg_scope.

      weight_decay: The weight decay to use for regularizing the model.
      stddev: Standard deviation for initialization, if negative uses xavier.
      dropout_keep_prob: dropout keep probability (not set if equals to None).
      bn_decay: decay for the batch norm moving averages (not set if equals to
        None).

    Returns:
      An argument scope to use via arg_scope.
    """
    # Note: do not introduce parameters that would change the inference
    # model here (for example whether to use bias), modify conv_def instead.
    batch_norm_params = {
        'decay': bn_decay,
        'is_training': is_training,
        'fused': False,
        'epsilon': 1e-5,
        'scale': True,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }
    if stddev < 0:
        weight_intitializer = slim.initializers.xavier_initializer()
    else:
        weight_intitializer = tf.truncated_normal_initializer(stddev=stddev)

    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected, slim.separable_conv2d],
            weights_initializer=weight_intitializer,
            normalizer_fn=slim.batch_norm), \
         safe_arg_scope([slim.batch_norm], **batch_norm_params), \
         safe_arg_scope([slim.dropout], is_training=is_training, keep_prob=dropout_keep_prob), \
         slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(weight_decay)), \
         slim.arg_scope([slim.separable_conv2d], weights_regularizer=None) as s:
        return s


def format_input(inputs):
    if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) == 1:
        inputs = inputs[0]
    return inputs


def _build_out_node(node, node_index, node_inputs, builder):
    params = node['module_params']
    sampled = int(node['sampled'])

    if len(params['module_list']) == 1:
        sampled_name = params['name_list'][0]
        sampled_module_params = params[params['name_list'][0]]
    else:
        sampled_name = params['name_list'][sampled]
        sampled_module_params = params[params['name_list'][sampled]]

    sampled_module_params.update({'scope': '%s_%s' % (sampled_name, str(node_index))})

    if len(params['module_list']) == 1:
        if sampled == 1:
            return builder()(node_inputs, **sampled_module_params)
        else:
            return None
    else:
        return builder()(node_inputs, **sampled_module_params)


class NasFactory(object):
    def __init__(self):
        pass

    def __getattr__(self, item):
        return globals()[item]()

    def _build_node(self, node, node_index, node_name, node_inputs):
        params = node['module_params']
        sampled = int(node['sampled'])

        if len(params['module_list']) == 1:
            sampled_module = params['module_list'][0]
            sampled_name = params['name_list'][0]
            sampled_module_params = params[params['name_list'][0]]
        else:
            sampled_module = params['module_list'][sampled]
            sampled_name = params['name_list'][sampled]
            sampled_module_params = params[params['name_list'][sampled]]

        sampled_module_params.update({'scope': '%s_%s' % (sampled_name, str(node_index))})

        # if node_index > 32:
        #     sampled_module_params.update({'rate': 2})

        if len(params['module_list']) == 1:
            if sampled == 1:
                return getattr(self, sampled_module)(node_inputs, **sampled_module_params)
            else:
                return None
        else:
            return getattr(self, sampled_module)(node_inputs, **sampled_module_params)

    def build(self, inputs, out_layers, architecture_path):
        graph = nx.read_gpickle(architecture_path)
        travel = list(nx.topological_sort(graph))

        input_feature = [inputs]
        layers_map = {}

        model_out = {}
        for node_index, node_name in enumerate(travel):
            print('build node %s TF op' % node_name)
            cur_node = graph.node[node_name]

            for pre_name in graph.predecessors(node_name):
                if layers_map[pre_name] is not None:
                    input_feature.append(layers_map[pre_name])

            input_feature = format_input(input_feature)
            if 'out' in cur_node['module_params']:
                outputs = _build_out_node(cur_node, node_index, input_feature,
                                          out_layers[cur_node['module_params']['out']])
            else:
                outputs = self._build_node(cur_node, node_index, node_name, input_feature)

            layers_map[node_name] = outputs
            input_feature = []

            if 'out' in cur_node['module_params']:
                model_out[cur_node['module_params']['out']] = outputs
        return model_out
