# @Time    : 2020/5/29 10:16
# @Author  : zhangchenming
import networkx as nx
from antnas.networks.SuperNetwork import SuperNetwork
from antnas.utils.drawers.NASDrawer import NASDrawer
from antnas.component.NetworkBlock import *


class FixSegArc:
    def __init__(self, in_chan, graph):
        self.graph = graph
        if self.graph is None:
            self.graph = nx.DiGraph()

        self.blocks = nn.ModuleList([])
        self.in_chan = in_chan
        self.hierarchical = [[[]]]

    def generate(self):
        in_name = SuperNetwork._INPUT_NODE_FORMAT.format(0, 0)

        head = ConvBn(self.in_chan, 32, k_size=3, stride=2, relu=True)

        self.graph.add_node(in_name,
                            module=len(self.blocks),
                            module_params=head.params,
                            sampling_param=len(self.blocks),
                            structure_fixed=head.structure_fixed,
                            pos=NASDrawer.get_draw_pos(pos=(0, 0)),
                            sampled=1)
        self.blocks.append(head)

        modules = [
            Fused(in_chan=head.params['out_chan'], out_chan=16, expand_factor=1, relu=True, k_size=3, stride=1,
                  dilation=1),
            Fused(in_chan=16, out_chan=32, expand_factor=8, relu=True, k_size=3, stride=2, dilation=1),
            Fused(in_chan=32, out_chan=32, expand_factor=4, relu=True, k_size=3, stride=1, dilation=1),
            Fused(in_chan=32, out_chan=32, expand_factor=4, relu=True, k_size=3, stride=1, dilation=1),
            Fused(in_chan=32, out_chan=32, expand_factor=4, relu=True, k_size=3, stride=1, dilation=1),

            Fused(in_chan=32, out_chan=48, expand_factor=8, relu=True, k_size=3, stride=2, dilation=1),
            Fused(in_chan=48, out_chan=48, expand_factor=4, relu=True, k_size=3, stride=1, dilation=1),
            Fused(in_chan=48, out_chan=48, expand_factor=4, relu=True, k_size=3, stride=1, dilation=1),
            Fused(in_chan=48, out_chan=48, expand_factor=4, relu=True, k_size=3, stride=1, dilation=1),

            InvertedResidualBlockWithSEHS(
                in_chan=48, out_chan=96, kernel_size=3, reduction=True, hs=False, se=False, expansion=8, skip=True),
            InvertedResidualBlockWithSEHS(
                in_chan=96, out_chan=96, kernel_size=3, reduction=False, hs=False, se=False, expansion=4, skip=True),
            InvertedResidualBlockWithSEHS(
                in_chan=96, out_chan=96, kernel_size=3, reduction=False, hs=False, se=False, expansion=4, skip=True),
            InvertedResidualBlockWithSEHS(
                in_chan=96, out_chan=96, kernel_size=3, reduction=False, hs=False, se=False, expansion=4, skip=True),

            InvertedResidualBlockWithSEHS(
                in_chan=96, out_chan=96, kernel_size=3, reduction=True, hs=False, se=False, expansion=8, skip=False),
            InvertedResidualBlockWithSEHS(
                in_chan=96, out_chan=96, kernel_size=3, reduction=False, hs=False, se=False, expansion=4, skip=True),
            InvertedResidualBlockWithSEHS(
                in_chan=96, out_chan=96, kernel_size=3, reduction=False, hs=False, se=False, expansion=4, skip=True),
            InvertedResidualBlockWithSEHS(
                in_chan=96, out_chan=96, kernel_size=3, reduction=False, hs=False, se=False, expansion=4, skip=True),

            InvertedResidualBlockWithSEHS(
                in_chan=96, out_chan=160, kernel_size=5, reduction=False, hs=False, se=False, expansion=8, skip=True),
            InvertedResidualBlockWithSEHS(
                in_chan=160, out_chan=160, kernel_size=3, reduction=False, hs=False, se=False, dilation=2, expansion=4,
                skip=True),
            InvertedResidualBlockWithSEHS(
                in_chan=160, out_chan=160, kernel_size=3, reduction=False, hs=False, se=False, dilation=2, expansion=4,
                skip=True),
            InvertedResidualBlockWithSEHS(
                in_chan=160, out_chan=160, kernel_size=3, reduction=False, hs=False, se=False, dilation=2, expansion=4,
                skip=True),

            InvertedResidualBlockWithSEHS(
                in_chan=160, out_chan=192, kernel_size=3, reduction=False, hs=False, se=False, dilation=2, expansion=8,
                skip=True),
        ]

        for module_index, module in enumerate(modules, start=1):
            module_index += 1
            module_name = SuperNetwork._FIXED_NODE_FORMAT.format(0, module_index)
            self.graph.add_node(module_name,
                                module=len(self.blocks),
                                module_params=module.params,
                                sampling_param=len(self.blocks),
                                structure_fixed=module.structure_fixed,
                                pos=NASDrawer.get_draw_pos(pos=(0, module_index)),
                                sampled=1)
            self.blocks.append(module)
            if module_index > 1:
                # 固定连接
                self.graph.add_edge(SuperNetwork._FIXED_NODE_FORMAT.format(0, module_index-1),
                                    SuperNetwork._FIXED_NODE_FORMAT.format(0, module_index),
                                    width_node=SuperNetwork._FIXED_NODE_FORMAT.format(0, module_index))

        self.graph.add_edge(SuperNetwork._INPUT_NODE_FORMAT.format(0, 0),
                            SuperNetwork._FIXED_NODE_FORMAT.format(0, 1),
                            width_node=SuperNetwork._FIXED_NODE_FORMAT.format(0, 1))

        offset = len(modules) + 1
        module = ASPPBlock(in_chan=192, depth=128, atrous_rates=[3, 6, 9])
        module_name = SuperNetwork._FIXED_NODE_FORMAT.format(0, offset)
        self.graph.add_node(module_name,
                            module=len(self.blocks),
                            module_params=module.params,
                            sampling_param=len(self.blocks),
                            structure_fixed=module.structure_fixed,
                            pos=NASDrawer.get_draw_pos(pos=(0, offset)),
                            sampled=1)

        self.graph.add_edge(SuperNetwork._FIXED_NODE_FORMAT.format(0, offset-1),
                            SuperNetwork._FIXED_NODE_FORMAT.format(0, offset),
                            width_node=SuperNetwork._FIXED_NODE_FORMAT.format(0, offset))

        offset = offset + 1
        module = ResizedBlock(in_chan=128, out_chan=2, scale_factor=16)
        out_name = SuperNetwork._FIXED_NODE_FORMAT.format(0, offset)
        self.graph.add_node(out_name,
                            module=len(self.blocks),
                            module_params=module.params,
                            sampling_param=len(self.blocks),
                            structure_fixed=module.structure_fixed,
                            pos=NASDrawer.get_draw_pos(pos=(0, offset)),
                            sampled=1)

        self.graph.add_edge(SuperNetwork._FIXED_NODE_FORMAT.format(0, offset-1),
                            SuperNetwork._FIXED_NODE_FORMAT.format(0, offset),
                            width_node=SuperNetwork._FIXED_NODE_FORMAT.format(0, offset))

        return in_name, out_name