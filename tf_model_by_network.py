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

def skip(inputs, out_channels, reduction):
    input_shape = inputs.get_shape()
    if out_channels > input_shape[-1]:
        pad = tf.zeros((input_shape[0], input_shape[1], input_shape[2], out_channels - input_shape[-1]))
        inputs = tf.concat([inputs, pad], axis=-1)

    if reduction:
        inputs = slim.avg_pool2d(inputs, kernel_size=2, stride=2, padding='VALID')

    return inputs


def inverted_residual_block_withse(inputs, expansion, kernel_size, out_chan, skip, reduction, ratio):
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
    print(se_x.get_shape())
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


def cell_block(inputs, out_channels, reduction, sampled):
    if sampled == 0:
        return skip(inputs, out_channels, reduction)
    elif sampled == 1:
        return inverted_residual_block_withse(inputs, expansion=3, kernel_size=3, out_chan=out_channels,
                                              skip=True, reduction=reduction, ratio=4)
    elif sampled == 2:
        return inverted_residual_block_withse(inputs, expansion=3, kernel_size=5, out_chan=out_channels,
                                              skip=True, reduction=reduction, ratio=4)
    elif sampled == 3:
        return inverted_residual_block_withse(inputs, expansion=6, kernel_size=3, out_chan=out_channels,
                                              skip=True, reduction=reduction, ratio=4)
    elif sampled == 4:
        return inverted_residual_block_withse(inputs, expansion=6, kernel_size=5, out_chan=out_channels,
                                              skip=True, reduction=reduction, ratio=4)
    else:
        raise ValueError('cell state %d do not exist' % sampled)


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


def get_operate_fn(node_name, node):
    params = node['module_params']
    print(params)

    def network_fn(inputs):
        if node_name.startswith('I'):
            return slim.conv2d(inputs,
                               num_outputs=params['out_chan'],
                               kernel_size=params['k_size'],
                               stride=params['stride'],
                               activation_fn=tf.nn.relu,
                               normalizer_fn=slim.batch_norm)

        elif node_name.startswith('A'):
            if not isinstance(inputs, list):
                return inputs
            assert isinstance(inputs, list)
            return sum(inputs)

        elif node_name.startswith('CELL'):
            return cell_block(inputs, out_channels=params['out_chan'],
                              reduction=params['reduction'], sampled=node['sampled'])

        elif node_name.startswith('T'):
            if node['module_type'] == 'conv':
                return conv_transfer_block(inputs, params['out_chan']) * float(node['sampled'] == 1)
            elif node['module_type'] == 'identity':
                return inputs * float(node['sampled'] == 1)

        elif node_name.startswith('O'):
            return out_layer(inputs, params['out_shape'])

        else:
            raise ValueError('Name of operate unknown %s' % node_name)

    return network_fn


def format_input(inputs):
    if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) == 1:
        inputs = inputs[0]
    return inputs


def get_nodel(images):
    graph = nx.read_gpickle("test.gpickle")
    travel = list(nx.topological_sort(graph))

    input_feature = [images]
    layers_map = {}
    i = 0
    for node_name in travel:
        print(node_name)
        cur_node = graph.node[node_name]
        print(cur_node)
        for pre_name in graph.predecessors(node_name):
            input_feature.append(layers_map[pre_name])

        input_feature = format_input(input_feature)

        output = get_operate_fn(node_name, cur_node)(input_feature)
        print(output.get_shape())
        layers_map[node_name] = output
        input_feature = []
        i += 1
        # if i == 3:
        #     break


if __name__ == '__main__':
    image = tf.random_uniform((1, 32, 32, 3))
    get_nodel(image)
