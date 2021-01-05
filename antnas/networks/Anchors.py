# -*- coding: UTF-8 -*-
# @Time    : 2020-04-07 15:35
# @File    : Anchors.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antnas.component.NetworkCell import *
from antnas.utils.drawers.NASDrawer import NASDrawer
from antnas.component.Loss import *
from antnas.component.ClassificationAccuracyEvaluator import *
from antnas.searchspace.StageBlockCellArc import *
from antnas.networks.FixedNetwork import *
from antnas.networks.FrozenFixedNetwork import *
import collections


class Anchors:
    def __init__(self, *args, **kwargs):
        self.anchor_archs = []
        self.anchor_index = 0
        self.anchor_num = 0

        self.ffn = []
        self.arc_list = []
        self.device_map = {}
        self.ffn_out = {}
    
    def generate(self, arch_generator, input_shape, constraint, arc_loss, folder, **kwargs):
        # 0.step build work folder
        if not os.path.exists(folder):
            os.makedirs(folder)

        # 1.step generate supernetwork
        arch_generator.generate(**kwargs)
        traversal_order = list(nx.topological_sort(arch_generator.graph))

        # 2.step find max/min arch
        max_loss = -1
        min_loss = -1
        try_times = 10

        while try_times > 0:
            feature = [None for _ in range(len(traversal_order))]
            for node_name in traversal_order:
                cur_node = arch_generator.graph.node[node_name]
                if not (node_name.startswith('CELL') or node_name.startswith('T')):
                    # 不可学习，处于永远激活状态
                    feature[cur_node['sampling_param']] = int(1)
                else:
                    if not arch_generator.blocks[cur_node['module']].structure_fixed:
                        feature[cur_node['sampling_param']] = int(np.random.randint(0, 2))
                    else:
                        feature[cur_node['sampling_param']] = int(np.random.randint(0, NetworkBlock.state_num))
        
            _, pruned_cost = \
                arch_generator.arc_loss(input_shape, loss=arc_loss, feature=feature)

            if max_loss < pruned_cost or max_loss < 0:
                max_loss = pruned_cost.item()
            
            if min_loss > pruned_cost or min_loss < 0:
                min_loss = pruned_cost.item()
            
            try_times -= 1
            
        # 3.step generate anchor arch
        if constraint is not None:
            constraint = [min_loss/max_loss] + constraint + [1.0]
        
        anchor_num = len(constraint) - 1
        for anchor_index in range(anchor_num):
            accept_min = min_loss
            accept_max = max_loss
            if constraint is not None:
                accept_min = constraint[anchor_index]
                accept_max = constraint[anchor_index+1]

            is_accept = False
            while not is_accept:
                # random sampling
                feature = [None for _ in range(len(traversal_order))]
                for node_name in traversal_order:
                    cur_node = arch_generator.graph.node[node_name]
                    sampled_val = 0
                    if not (node_name.startswith('CELL') or node_name.startswith('T')):
                        # 不可学习，处于永远激活状态
                        sampled_val = int(1)
                    else:
                        if not arch_generator.blocks[cur_node['module']].structure_fixed:
                            sampled_val = int(np.random.randint(0, 2))
                        else:
                            sampled_val = int(np.random.randint(0, NetworkBlock.state_num))

                    arch_generator.graph.node[node_name]['sampled'] = sampled_val
                    feature[cur_node['sampling_param']] = sampled_val

                sampled_cost, pruned_cost = \
                    arch_generator.arc_loss(input_shape, loss=arc_loss, feature=feature)

                if not (pruned_cost / max_loss> accept_min and pruned_cost / max_loss <= accept_max):
                    continue

                architecture_path = \
                    os.path.join(folder,
                                 "anchor_arch_%d_%0.1f%%_%0.1f%%.architecture" % (anchor_index,
                                                                                  accept_min*100,
                                                                                  accept_max*100))
                nx.write_gpickle(arch_generator.graph, architecture_path)
                self.anchor_archs.append(os.path.join(folder,
                                                      "anchor_arch_%d_%0.1f%%_%0.1f%%.architecture" % (anchor_index,
                                                                                                       accept_min*100,
                                                                                                       accept_max*100)))
                is_accept = True
                
    def load(self, archs_list, states_list, output_layer_cls, cuda_device_list=None):
        for arch_index, arch_file in enumerate(archs_list):
            # build architecture
            self.ffn.append(FrozenFixedNetwork(architecture=arch_file, output_layer_cls=output_layer_cls))

            # reassign device
            if torch.cuda.is_available():
                self.ffn[arch_index].to(cuda_device_list[0])

                for index, cuda_device_id in enumerate(cuda_device_list):
                    self.device_map["cuda:%d"%cuda_device_id] = index
            else:
                self.device_map['cpu'] = 0

            # load parameter
            if torch.cuda.is_available():
                kv = torch.load(states_list[arch_index], map_location=torch.device("cuda:%d"%cuda_device_list[0]))
                new_kv = collections.OrderedDict()
                for k, v in kv.items():
                    new_kv['.'.join(k.split('.')[1:])] = v

                self.ffn[arch_index].load_state_dict(new_kv)
            else:
                kv = torch.load(states_list[arch_index], map_location="cpu")
                new_kv = collections.OrderedDict()
                for k, v in kv.items():
                    new_kv['.'.join(k.split('.')[1:])] = v

                self.ffn[arch_index].load_state_dict(new_kv)

            # record arch
            feature = []
            for node_name in self.ffn[arch_index].traversal_order:
                feature.append(int(self.ffn[arch_index].graph.node[node_name]['sampled']))

            self.arc_list.append(feature)

            # param no grad
            for param in self.ffn[arch_index].parameters():
                param.requires_grad = False

            # is parallel
            if torch.cuda.is_available():
                self.ffn[arch_index] = nn.DataParallel(self.ffn[arch_index], cuda_device_list)

        self.anchor_num = len(archs_list)

    def arch(self, anchor_index):
        return self.arc_list[anchor_index]

    def run(self, x, y):
        self.ffn_out = {}
        for anchor_index in range(self.anchor_num):
            self.ffn[anchor_index].eval()
            with torch.no_grad():
                self.ffn_out[anchor_index] = self.ffn[anchor_index](x, y)

    def output(self, anchor_index, node_name):
        output = self.ffn_out[anchor_index][node_name]
        return output

    def size(self):
        return self.anchor_num