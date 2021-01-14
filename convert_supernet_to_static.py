# -*- coding: UTF-8 -*-
# @Time    : 2021/1/13 5:20 下午
# @File    : convert_supernet_to_static.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import networkx as nx
import copy
import argparse
import tensorflow as tf
from antnas.tf.builder.nasblock import *
from tensorflow.python.framework import graph_util


parser = argparse.ArgumentParser()
parser.add_argument('--architecture', type=str, default='')
parser.add_argument('--shape', default='1,224,224,3', type=str)


def format_input(input):
    if (isinstance(input, tuple) or isinstance(input, list)) and len(input) == 1:
        input = input[0]
    return input


if __name__ == '__main__':
    args = parser.parse_args()
    # model architecture
    architecture_path = args.architecture
    shape = args.shape
    input_shape = [(int)(s) for s in shape.split(',')]

    graph = nx.read_gpickle(architecture_path)
    traversal_order = list(nx.topological_sort(graph))

    in_node = None
    out_node = None
    data_dict = {}
    input = tf.placeholder(dtype=tf.float32, shape=input_shape)
    out = None
    with tf.Session() as sess:
        for node_index, node_name in enumerate(traversal_order):
            print(node_name)
            if node_index == 0:
                in_node = node_name
                data_dict[in_node] = format_input([input])
            if node_index == len(traversal_order) - 1:
                out_node = node_name
                break

            input = format_input(data_dict[node_name])

            cur_node = graph.node[node_name]
            module_list = cur_node['module_params']['module_list']
            name_list = cur_node['module_params']['name_list']
            module_outs = []
            for model, model_name in zip(module_list, name_list):
                model_param = cur_node['module_params'][model_name]
                model_obj = globals()[model]()
                model_out = model_obj(input, scope='%s-%d-%s'%(node_name, node_index, model_name), **model_param)
                module_outs.append(model_out)

            # 加入Add算子合并model输出
            out = tf.add_n(module_outs)

            #
            for succ in graph.successors(node_name):
                if succ not in data_dict:
                    data_dict[succ] = []
                data_dict[succ].append(out)

        out = tf.identity(out, name='out')

        sess.run(tf.initialize_all_variables())
        # We use a built-in TF helper to export variables to constant
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            sess.graph.as_graph_def(),
            ['out']
        )

        with tf.gfile.GFile('./graph.pb', "wb") as f:
            f.write(output_graph_def.SerializeToString())
