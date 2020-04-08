# -*- coding: UTF-8 -*-
# @Time    : 2020-04-07 15:35
# @File    : Anchors.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from nas.component.NetworkCell import *
from nas.networks.EvolutionSuperNetwork import *
from nas.utils.drawers.BSNDrawer import BSNDrawer
from nas.component.Loss import *
from nas.component.ClassificationAccuracyEvaluator import *
from nas.searchspace.StageBlockCellArc import *
from nas.networks.FixedNetwork import *
from nas.networks.FrozenFixedNetwork import *
import collections

class Anchors:
    def __init__(self, *args, **kwargs):
        self.anchor_archs = []
        self.anchor_index = 0
        self.anchor_num = 0

        self.ffn = []
        self.arc_list = []

    def generate(self, arch_generator, anchor_num, folder, **kwargs):
        # 0.step build work folder
        if not os.path.exists(folder):
            os.makedirs(folder)

        # 1.step generate supernetwork
        arch_generator.generate(**kwargs)

        # 2.step generate anchor arch
        traversal_order = list(nx.topological_sort(arch_generator.graph))
        for anchor_index in range(anchor_num):
            # random sampling
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

            architecture_path = os.path.join(folder, "anchor_arch_%s.architecture" % anchor_index)
            nx.write_gpickle(arch_generator.graph, architecture_path)
            self.anchor_archs.append(os.path.join(folder, "anchor_arch_%s.architecture" % anchor_index))

    def load(self, archs_list, states_list, output_layer_cls, cuda_device=None):
        # 1.step 创建结构
        for arch_file in archs_list:
            self.ffn.append(FrozenFixedNetwork(architecture=arch_file, output_layer_cls=output_layer_cls))
        self.anchor_num = len(archs_list)

        for anchor_index in range(self.anchor_num):
            feature = []
            for node_name in self.ffn[anchor_index].traversal_order:
                feature.append(int(self.ffn[anchor_index].graph.node[node_name]['sampled']))

            self.arc_list.append(feature)

        # 2.step 迁移至GPU
        if torch.cuda.is_available():
            for anchor_index in range(self.anchor_num):
                self.ffn[anchor_index].to(cuda_device)

        # 3.step 加载参数
        for anchor_index in range(self.anchor_num):
            if torch.cuda.is_available():
                kv = torch.load(states_list[anchor_index], map_location=cuda_device)
                new_kv = collections.OrderedDict()
                for k,v in kv.items():
                    new_kv['.'.join(k.split('.')[1:])] = v

                self.ffn[anchor_index].load_state_dict(new_kv)
            else:
                kv = torch.load(states_list[anchor_index], map_location="cpu")
                new_kv = collections.OrderedDict()
                for k, v in kv.items():
                    new_kv['.'.join(k.split('.')[1:])] = v

                self.ffn[anchor_index].load_state_dict(new_kv)

    def arch(self, anchor_index):
        return self.arc_list[anchor_index]

    def run(self, x, y):
        for anchor_index in range(self.anchor_num):
            self.ffn[anchor_index].eval()
            with torch.no_grad():
                self.ffn[anchor_index](x, y)

    def output(self, anchor_index, node_name):
        return self.ffn[anchor_index].graph.node[node_name]['out']

    def size(self):
        return self.anchor_num