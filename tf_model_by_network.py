# @Time    : 2019-08-23 10:42
# @Author  : zhangchenming
import networkx as nx
import tensorflow as tf

slim = tf.contrib.slim


# graph = nx.DiGraph()
# graph.add_node('first', module='conv2d', stride=2)
# graph.add_node('next', module='conv2d', stride=2, atrous=2)
# graph.add_edge('first', 'next')
# nx.write_gpickle(graph, "test.gpickle")

def convbn(inputs, out_chan, k_size, stride):
    return slim.conv2d(inputs,
                       num_outputs=out_chan,
                       kernel_size=k_size,
                       stride=stride,
                       activation_fn=tf.nn.relu,
                       normalizer_fn=slim.batch_norm)


def add_block(inputs):
    if not isinstance(inputs, list):
        return inputs
    assert isinstance(inputs, list)
    return sum(inputs)


def skip(inputs, out_chan, reduction):
    input_shape = inputs.get_shape()
    if out_chan > input_shape[-1]:
        pad = tf.zeros((input_shape[0], input_shape[1], input_shape[2], out_chan - input_shape[-1]))
        inputs = tf.concat([inputs, pad], axis=-1)

    if reduction:
        inputs = slim.avg_pool2d(inputs, kernel_size=2, stride=2, padding='VALID')

    return inputs


def inverted_residual_block_withse(inputs, expansion, kernel_size, out_chan, reduction, skip=True, ratio=4):
    x = inputs
    input_shape = inputs.get_shape().as_list()
    x = slim.conv2d(x,
                    num_outputs=input_shape[3] * expansion,
                    kernel_size=1,
                    stride=1,
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm)
    x = slim.separable_conv2d(x,
                              num_outputs=None,
                              kernel_size=kernel_size,
                              stride=2 if reduction else 1,
                              activation_fn=tf.nn.relu,
                              normalizer_fn=slim.batch_norm)
    se_x = tf.reduce_mean(x, axis=[1, 2], keep_dims=False)
    se_x = slim.fully_connected(se_x, num_outputs=input_shape[3] * expansion // ratio, activation_fn=tf.nn.relu)
    se_x = slim.fully_connected(se_x, num_outputs=input_shape[3] * expansion, activation_fn=None)
    se_x = (0.2 * se_x) + 0.5
    se_x = tf.clip_by_value(se_x, 0.0, 1.0)
    se_x = tf.reshape(se_x, shape=[input_shape[0], 1, 1, input_shape[3] * expansion])
    x = tf.multiply(se_x, x)
    x = tf.nn.relu(x)
    x = slim.conv2d(x,
                    num_outputs=out_chan,
                    kernel_size=1,
                    stride=1,
                    activation_fn=None,
                    normalizer_fn=slim.batch_norm)
    if skip and input_shape[0] == out_chan and (not reduction):
        x = x + inputs
    return x


def conv_transfer_block(inputs, out_chan):
    x = slim.conv2d(inputs,
                    num_outputs=out_chan,
                    kernel_size=1,
                    stride=1,
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm)
    return x


def out_layer(inputs, out_shape):
    input_shape = inputs.get_shape().as_list()
    x = slim.conv2d(inputs,
                    num_outputs=input_shape[-1],
                    kernel_size=1,
                    stride=1,
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm)
    x = slim.avg_pool2d(x, kernel_size=2, stride=2, padding='VALID')
    x = slim.conv2d(x,
                    num_outputs=out_shape[0],
                    kernel_size=1,
                    stride=1,
                    activation_fn=None,
                    normalizer_fn=None)
    x = tf.reshape(x, (-1, *out_shape))
    return x


def identity(inputs):
    return inputs


op_list = {
    'convbn': convbn,
    'add_block': add_block,
    'inverted_residual_block_withse': inverted_residual_block_withse,
    'identity': identity,
    'skip': skip,
    'conv_transfer_block': conv_transfer_block,
    'out_layer': out_layer
}

op_name_trans = {
    'convbn': 'convbn',
    'dummy_block': 'identity',
    'add_block': 'add_block',
    'IRB_k3e3_skip': 'inverted_residual_block_withse',
    'IRB_k5e3_skip': 'inverted_residual_block_withse',
    'IRB_k3e6_skip': 'inverted_residual_block_withse',
    'IRB_k5e6_skip': 'inverted_residual_block_withse',
    'skip': 'skip',
    'conv_transfer_block': 'conv_transfer_block',
    'out_layer': 'out_layer'
}


def get_operate_fn(node):
    params = node['module_params']
    # print(params)
    sampled = node['sampled']

    if len(params['module_list']) == 1:
        sampled_name = params['module_list'][0]
        op_params = params[sampled_name]
        op_name = op_name_trans[sampled_name]
    else:
        sampled_name = params['module_list'][sampled]
        op_params = params[sampled_name]
        op_name = op_name_trans[sampled_name]

    print(op_name)

    def network_fn(inputs):
        if len(params['module_list']) == 1:
            return op_list[op_name](inputs, **op_params) * float(sampled == 1)
        else:
            return op_list[op_name](inputs, **op_params)

    return network_fn


def format_input(inputs):
    if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) == 1:
        inputs = inputs[0]
    return inputs


def get_model(images):
    graph = nx.read_gpickle("test.gpickle")
    travel = list(nx.topological_sort(graph))

    input_feature = [images]
    layers_map = {}
    i = 0
    for node_name in travel:
        print(node_name)
        cur_node = graph.node[node_name]
        for pre_name in graph.predecessors(node_name):
            input_feature.append(layers_map[pre_name])

        input_feature = format_input(input_feature)

        outputs = get_operate_fn(cur_node)(input_feature)
        print(outputs.get_shape())
        layers_map[node_name] = outputs
        input_feature = []
        i += 1
        # if i == 3:
        #     break


if __name__ == '__main__':
    image = tf.random_uniform((1, 32, 32, 3))
    get_model(image)
