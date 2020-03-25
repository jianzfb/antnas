from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from nas.networks.SuperNetwork import SuperNetwork
from nas.component.NetworkBlock import *
from nas.component.NetworkCell import *
from nas.component.PathRecorder import PathRecorder
import copy
import networkx as nx
import threading
import pickle
from nas.networks.bayesian import *


class StochasticSuperNetwork(SuperNetwork):
    INIT_NODE_PARAM = 3

    def __init__(self, deter_eval, *args, **kwargs):
        super(StochasticSuperNetwork, self).__init__(*args, **kwargs)
        self.nodes_param = None
        self.probas = None
        self.entropies = None
        self.mean_entropy = None
        self.deter_eval = deter_eval

    def get_node_sampling(self, node_name, batch_size, batched_sampling):
        """
        Get a batch of sampling for the given node on the given output.
        Fires a "sampling" event with the node name and the sampling Variable.
        :param node_name: Name of the node to sample
        :param out: Tensor on which the sampling will be applied
        :return: A Variable brodcastable to out's size, with all dimensions equals to one except the first one (batch)
        """
        sampling_dim = [batch_size] + [1] * 3

        if self.use_preload_architecture:
            val = self.net.node[node_name]['sampled']
            node_sampling = torch.Tensor().resize_(*sampling_dim).fill_(val)
            node_sampling = Variable(node_sampling, requires_grad=False)
            return node_sampling

        static_sampling = self._get_node_static_sampling(node_name)
        if static_sampling is not None:
            node_sampling = torch.Tensor().resize_(*sampling_dim).fill_(static_sampling)
            node_sampling = Variable(node_sampling, requires_grad=False)
        else:
            node = self.net.node[node_name]
            node_sampling = batched_sampling[:, node['sampling_param']].contiguous().view(sampling_dim)

        # self.fire(type='sampling', node=node_name, value=sampling)
        return node_sampling

    def _get_node_static_sampling(self, node_name):
        """
        Method used to check if the sampling should be done or if the node is static.
        Raise error when used with an old version of the graph.
        :param node_name:
        :return: The value of the sampling (0 or 1) if the given node is static. None otherwise.

        """
        node = self.net.node[node_name]
        if 'sampling_val' in node or 'sampling_param' not in node or node['sampling_param'] is None:
            raise RuntimeError('Deprecated method. {} node has no attribute `sampling_param` or has attribute '
                               '`sampling_val`.All nodes should now have a param '
                               '(static ones are np.inf with non trainable param).'.format(node_name))

        if not self.training and self.deter_eval:
            # Model is in deterministic evaluation mode
            sampling_distribution = torch.nn.Softmax2d()(self.sampling_parameters[node['sampling_param']].view(1, NetworkBlock.state_num,1,1))
            sampling_distribution = torch.squeeze(sampling_distribution)
            preditions_argmax = sampling_distribution.argmax(0)
            return preditions_argmax

        return None

    def forward(self, *input):
        # assert
        assert len(input) == 2

        # 0.step copy network graph
        running_graph = copy.deepcopy(self.net)

        # 1.step set sampling and active on batch
        sampling = torch.Tensor()
        active = torch.Tensor()

        # 2.step parse x,y - (data,label)
        x, y = input
        input = [x]

        # 3.step get sampling architecture
        batched_sampling, batched_log_probas = self._sample_archs(input[0].size(0))

        # 4.step forward network
        # 4.1.step set the input of network graph
        running_graph.node[self.in_node]['input'] = [*input]
        model_out = None
        node_regulizer_loss = []
        for node in self.traversal_order:
            cur_node = running_graph.node[node]
            input = self.format_input(cur_node['input'])

            if len(input) == 0:
                raise RuntimeError('Node {} has no inputs'.format(node))

            batch_size = input[0].size(0) if type(input) == list else input.size(0)
            node_sampling = self.get_node_sampling(node, batch_size, batched_sampling)
            # notify path recorder to add sampling
            sampling, active = self.add_sampling(node, node_sampling, sampling, active, self.blocks[cur_node['module']].switch)

            # 4.2.step set node sampling
            self.blocks[cur_node['module']].set_sampling(node_sampling)

            # 4.3.step set node last sampling
            if self.last_sampling is not None:
                node_last_sampling = self.last_sampling[self.path_recorder.node_index[node]]
                self.blocks[cur_node['module']].set_last_sampling(node_last_sampling)

            # 4.4.step execute node op
            out = self.blocks[cur_node['module']](input)

            # 4.5.step add regularizer loss
            if self.blocks[cur_node['module']].get_node_regularizer() is not None:
                node_regulizer_loss.append(self.blocks[cur_node['module']].get_node_regularizer())

            if node == self.out_node:
                model_out = out
                break

            # 4.6.step set successor input
            for succ in running_graph.successors(node):
                if 'input' not in running_graph.node[succ]:
                    running_graph.node[succ]['input'] = []
                running_graph.node[succ]['input'].append(out)

        # 5.step notify path recorder to update global statistics
        self.update(sampling, active)

        # 6.step compute model loss
        indiv_loss = self.loss(model_out, y)

        # 7.step compute model accuracy
        model_accuracy = self.accuray(model_out, y)

        # 8.step compute architecture loss
        optim_cost = None
        sampled_cost = None
        pruned_cost = None
        for cost, cost_eval in self.architecture_cost_evaluators.items():
            sampled_cost, pruned_cost = cost_eval.get_costs(self.architecture(sampling, active), running_graph)

            if cost == self.architecture_cost_optimization:
                optim_cost = sampled_cost

        # cost = (optim_cost + self.architecture_penalty * self.architecture_consistence(sampling, active)) - self.architecture_objective_cost
        cost = optim_cost - self.architecture_objective_cost

        if self.architecture_objective_method == 'max':
            cost.clamp_(min=0)
        elif self.architecture_objective_method == 'abs':
            cost.abs_()

        cost = indiv_loss.data.new(cost.size()).copy_(cost)
        rewards = -(indiv_loss.data.squeeze() + self.architecture_lambda * cost)
        mean_reward = rewards.mean()
        rewards = (rewards - mean_reward)

        rewards = Variable(rewards)
        sampling_loss = self.architecture_loss(rewards=rewards, batched_log_probas=batched_log_probas)

        # 9.step compute regularizer loss
        regularizer_loss = 0.0
        if len(node_regulizer_loss) > 0 and self.kwargs['regularizer']:
            regularizer_loss = torch.Tensor(node_regulizer_loss).mean()

        # 10.step total loss
        loss = indiv_loss.mean() + sampling_loss + 0.001 * regularizer_loss

        sampled_cost_ = torch.as_tensor(sampled_cost, device=loss.device)
        pruned_cost_ = torch.as_tensor(pruned_cost, device=loss.device)
        return loss, model_accuracy, sampled_cost_, pruned_cost_

    def _sample_archs(self, batch_size):
        batch_size = int(batch_size)

        # params = torch.stack([p for p in self.sampling_parameters], dim=1)
        # probas_resized = params.sigmoid().expand(batch_size, len(self.sampling_parameters))
        # distrib = torch.distributions.Bernoulli(probas_resized)
        #
        # key_map = {}
        # for i, node in enumerate(self.graph.nodes()):
        #     key_map[node] = i
        #
        # params = torch.stack([self.sampling_parameters[key_map[order_name]] for order_name in self.traversal_order], dim=0)
        # TODO temperature
        params = torch.stack([p for p in self.sampling_parameters], dim=0)
        params = params.clamp(min=-1e+20, max=1e+20)
        probas_resized = params.softmax(dim=-1).expand(batch_size,
                                                       params.size(0),
                                                       params.size(1))
        distrib = torch.distributions.categorical.Categorical(probas_resized)

        batched_sampling = distrib.sample()
        batched_log_probas = distrib.log_prob(batched_sampling)
        return batched_sampling, batched_log_probas

    @property
    def n_layers(self):
        return sum([mod.n_layers for mod in self.blocks])

    @property
    def n_comp_steps(self):
        return sum([mod.n_comp_steps for mod in self.blocks])

    # def reinit_sampling_params(self):
    #     new_params = nn.ParameterList()
    #     for p in self.sampling_parameters:
    #         if p.requires_grad:
    #             param_value = self.INIT_NODE_PARAM
    #         else:
    #             param_value = p.data[0]
    #         new_params.append(nn.Parameter(p.data.new(([param_value])), requires_grad=p.requires_grad))
    #
    #     self.sampling_parameters = new_params

    # def update_probas_and_entropies(self):
    #     if self.nodes_param is None:
    #         self._init_nodes_param()
    #
    #     self.probas = {}
    #     self.entropies = {}
    #     self.mean_entropy = .0
    #     for node, props in self.graph.node.items():
    #         param = self.sampling_parameters[props['sampling_param']]
    #         p = param.sigmoid().item()
    #         self.probas[node] = p
    #         if p in [0, 1]:
    #             e = 0
    #         else:
    #             e = -(p * np.log2(p)) - ((1 - p) * np.log2(1 - p))
    #         self.entropies[node] = e
    #         self.mean_entropy += e
    #     self.mean_entropy /= self.graph.number_of_nodes()

    # def _init_nodes_param(self):
    #     self.nodes_param = {}
    #     for node, props in self.graph.node.items():
    #         if 'sampling_param' in props and props['sampling_param'] is not None:
    #             self.nodes_param[node] = props['sampling_param']

    def architecture_loss(self, *args, **kwargs):
        rewards = kwargs['rewards']
        batched_log_probas = kwargs['batched_log_probas']
        sampling_loss_val = (-batched_log_probas * rewards.unsqueeze(dim=1)).sum()
        return sampling_loss_val

    def save_architecture(self, folder=None, name=None):
        if not os.path.exists(folder):
            os.makedirs(folder)

        path = os.path.join(folder, name)
        # 1.step save model
        torch.save(self.state_dict(), '%s.model'%path)

        # 2.step save architecture
        # 2.0.step sampling network
        batched_sampling, _ = self._sample_archs(1)

        # 2.1.step prune sampling network
        sampling = torch.Tensor()
        active = torch.Tensor()

        for node in self.traversal_order:
            cur_node = self.net.node[node]
            node_sampling = self.get_node_sampling(node, 1, batched_sampling)

            # notify path recorder to add sampling
            sampling, active = self.add_sampling(node, node_sampling, sampling, active, self.blocks[cur_node['module']].switch)

        _, pruned_architecture = self.architecture(sampling, active)

        # 2.2.step write to graph
        for node in self.traversal_order:
            node_sampling_val = torch.squeeze(pruned_architecture[self.path_recorder.node_index[node]]).item()
            self.net.node[node]['sampled'] = int(node_sampling_val)

        # 2.3.step save architecture
        architecture_path = '%s.architecture'%path
        nx.write_gpickle(self.net, architecture_path)

        # 2.4.step save model weight
        def _extract_params_func(layer, prefix, sub_prefix, conv_count, depthconv_count, is_conv2d):
            has_childre_num = 0
            for _ in layer.children():
                has_childre_num += 1

            param_dict = {}
            if has_childre_num > 0:
                for sub_layer in layer.children():
                    local_param, prefix, conv_count, depthconv_count, is_conv2d = _extract_params_func(sub_layer, prefix, layer._get_name(), conv_count, depthconv_count, is_conv2d)
                    param_dict.update(local_param)
            else:
                if isinstance(layer, torch.nn.modules.conv.Conv2d):
                    if layer.groups > 1:
                        # depthwise conv
                        param_dict['%s/%s/SeparableConv2d%s/depthwise_weights' % (self.__class__.__name__,
                                                             prefix,
                                                             '_%d'%depthconv_count if depthconv_count > 0 else '')] = layer.weight.cpu().data.numpy().transpose(2, 3, 0, 1)

                        print('%s/%s/SeparableConv2d%s/depthwise_weights' % (self.__class__.__name__,
                                                             prefix,
                                                             '_%d'%depthconv_count if depthconv_count > 0 else ''))
                        print(param_dict['%s/%s/SeparableConv2d%s/depthwise_weights' % (self.__class__.__name__,
                                                             prefix,
                                                             '_%d'%depthconv_count if depthconv_count > 0 else '')].shape)

                        if layer.bias is not None:
                            param_dict['%s/%s/SeparableConv2d%s/biases' % (
                                self.__class__.__name__, prefix, '_%d'%depthconv_count if depthconv_count > 0 else '')] = layer.bias.cpu().data.numpy()

                        is_conv2d = False
                        depthconv_count += 1
                    else:
                        if sub_prefix.startswith('SepConvBN'):
                            # pointwise conv
                            depthconv_count = depthconv_count - 1
                            param_dict['%s/%s/SeparableConv2d%s/pointwise_weights' % (self.__class__.__name__,
                                                                                      prefix,
                                                                                      '_%d' % depthconv_count if depthconv_count > 0 else '')] = layer.weight.cpu().data.numpy().transpose(
                                2, 3, 1, 0)

                            print('%s/%s/SeparableConv2d%s/pointwise_weights' % (self.__class__.__name__,
                                                                                 prefix,
                                                                                 '_%d' % depthconv_count if depthconv_count > 0 else ''))
                            print(param_dict['%s/%s/SeparableConv2d%s/pointwise_weights' % (self.__class__.__name__,
                                                                                            prefix,
                                                                                            '_%d' % depthconv_count if depthconv_count > 0 else '')].shape)
                            is_conv2d = False
                            depthconv_count = depthconv_count + 1
                        else:
                            # conv
                            param_dict['%s/%s/Conv%s/weights' % (self.__class__.__name__,
                                                                 prefix,
                                                                 '_%d'%conv_count if conv_count > 0 else '')] = layer.weight.cpu().data.numpy().transpose(2, 3, 1, 0)

                            print('%s/%s/Conv%s/weights' % (self.__class__.__name__,
                                                                 prefix,
                                                                 '_%d'%conv_count if conv_count > 0 else ''))
                            print(param_dict['%s/%s/Conv%s/weights' % (self.__class__.__name__,
                                                                 prefix,
                                                                 '_%d'%conv_count if conv_count > 0 else '')].shape)
                            if layer.bias is not None:
                                param_dict['%s/%s/Conv%s/biases' % (
                                    self.__class__.__name__, prefix, '_%d'%conv_count if conv_count > 0 else '')] = layer.bias.cpu().data.numpy()

                            is_conv2d = True
                            conv_count += 1
                elif isinstance(layer, torch.nn.modules.batchnorm.BatchNorm2d):
                    if is_conv2d:
                        conv_count = conv_count - 1
                        param_dict['%s/%s/Conv%s/BatchNorm/moving_mean' % (self.__class__.__name__,
                                                                           prefix,
                                                                           '_%d'%conv_count if conv_count > 0 else '')] = layer.running_mean.cpu().numpy()
                        param_dict['%s/%s/Conv%s/BatchNorm/moving_variance' % (self.__class__.__name__,
                                                                               prefix,
                                                                               '_%d'%conv_count if conv_count > 0 else '')] = layer.running_var.cpu().numpy()
                        param_dict['%s/%s/Conv%s/BatchNorm/gamma' % (self.__class__.__name__,
                                                                     prefix,
                                                                     '_%d'%conv_count if conv_count > 0 else '')] = layer.weight.cpu().data.numpy()
                        param_dict['%s/%s/Conv%s/BatchNorm/beta' % (self.__class__.__name__,
                                                                    prefix,
                                                                    '_%d'%conv_count if conv_count > 0 else '')] = layer.bias.cpu().data.numpy()
                        conv_count = conv_count + 1
                    else:
                        depthconv_count = depthconv_count - 1
                        param_dict['%s/%s/SeparableConv2d%s/BatchNorm/moving_mean' % (self.__class__.__name__,
                                                                           prefix,
                                                                           '_%d'%depthconv_count if depthconv_count > 0 else '')] = layer.running_mean.cpu().numpy()
                        param_dict['%s/%s/SeparableConv2d%s/BatchNorm/moving_variance' % (self.__class__.__name__,
                                                                               prefix,
                                                                               '_%d'%depthconv_count if depthconv_count > 0 else '')] = layer.running_var.cpu().numpy()
                        param_dict['%s/%s/SeparableConv2d%s/BatchNorm/gamma' % (self.__class__.__name__,
                                                                     prefix,
                                                                     '_%d'%depthconv_count if depthconv_count > 0 else '')] = layer.weight.cpu().data.numpy()
                        param_dict['%s/%s/SeparableConv2d%s/BatchNorm/beta' % (self.__class__.__name__,
                                                                    prefix,
                                                                    '_%d'%depthconv_count if depthconv_count > 0 else '')] = layer.bias.cpu().data.numpy()

                        depthconv_count = depthconv_count + 1
                else:
                    print('why im here')
                    print(layer)

            return param_dict, prefix, conv_count, depthconv_count, is_conv2d

        param_dict = {}
        for node_index, node_name in enumerate(self.traversal_order):
            node = self.net.node[node_name]
            node_params = self.blocks[node['module']].params
            node_sampled = node['sampled']
            prefix = ''
            if len(node_params['module_list']) == 1:
                prefix = '%s_%s'%(node_params['name_list'][0], str(node_index))
            else:
                prefix = '%s_%s'%(node_params['name_list'][node_sampled], str(node_index))

            if prefix.startswith('GCN'):
                print(node_name)

            models_sampled = []
            if len(node_params['module_list']) == 1:
                models_sampled = self.blocks[self.net.node[node_name]['module']]
            else:
                models_sampled = self.blocks[self.net.node[node_name]['module']].op_list[node_sampled]

            mm, _, _, _, _ = _extract_params_func(models_sampled, prefix, '', 0, 0, False)
            param_dict.update(mm)

        f = open('%s.weight'%path, "wb")
        pickle.dump(param_dict, f)
        f.close()

    def sampling_param_generator(self, node_name):
        if not (node_name.startswith('CELL') or node_name.startswith('T')):
            # 不可学习，处于永远激活状态
            param_value = [0] + [1000000000000000] + [0] * (CellBlock.state_num - 2)
            trainable = False
        else:
            param_value = [0] + [np.log(0.9526 * (CellBlock.state_num - 1) / 0.0474)] + [0] * (CellBlock.state_num - 2)
            trainable = True

        # param_value = [0] + [1000000000000000] + [0] * (CellBlock.state_num - 2)
        # trainable = False
        return nn.Parameter(torch.Tensor(param_value), requires_grad=trainable)

    def plot(self, path=None):
        pass

    def __str__(self):
        model_descr = 'Model:{}\n\t{} nodes\n\t{} blocks\n\t{} parametrized layers\n\t{} computation steps\n\t{} parameters\n\t{} meta-params'
        return model_descr.format(type(self).__name__,
                                  self.graph.number_of_nodes(),
                                  len(self.blocks),
                                  self.n_layers,
                                  self.n_comp_steps,
                                  sum(i.numel() for i in self.parameters()),
                                  len(self.sampling_parameters))
