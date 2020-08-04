# -*- coding: UTF-8 -*-
# @Time    : 2020-05-15 19:05
# @File    : PKMixArc.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antnas.searchspace.Arc import *
from antnas.searchspace.PKAutoArc import *
from antnas.component.NetworkBlock import *
from antnas.component.NetworkBlock import _make_divisible


class PKMixArc(PKAutoArc):
    def __init__(self,
                 cell_cls,
                 aspp_cell_cls,
                 aggregation_cls,
                 transformer_cls,
                 graph,
                 blocks,
                 sampling_param_generator=None,
                 backbone='',
                 decoder_input_endpoints=[0,2,5,16],
                 decoder_input_strides=[4,8,16,32],
                 decoder_depth=32,
                 decoder_replica=1,
                 decoder_allow_skip=False):
        super(PKMixArc, self).__init__(graph, blocks)
        self.sampling_param_generator = sampling_param_generator
        self.sampling_parameters = nn.ParameterList()

        self.cell_cls = cell_cls
        self.aspp_cell_cls = aspp_cell_cls
        self.transformer_cls = transformer_cls
        self.aggregation_cls = aggregation_cls
        self.hierarchical = [[[]]]
        self.backbone = backbone
        self.decoder_input_endpoints = decoder_input_endpoints
        self.decoder_input_strides = decoder_input_strides
        self.decoder_depth = decoder_depth
        self.decoder_allow_skip = decoder_allow_skip
        self.decoder_replica = decoder_replica
    
    def add_fixed(self, pos, module):
        node_name = SuperNetwork._FIXED_NODE_FORMAT.format(*pos)
        sampling_param = None
        if self.sampling_param_generator is not None:
            sampling_param = self.sampling_param_generator(node_name)
    
        self.graph.add_node(node_name,
                            module=len(self.blocks),
                            module_params=module.params,
                            sampling_param=len(self.blocks),
                            structure_fixed=module.structure_fixed,
                            pos=NASDrawer.get_draw_pos(pos=pos))
    
        if sampling_param is not None:
            self.sampling_parameters.append(sampling_param)
    
        self.blocks.append(module)
        return node_name

    def add_cell(self, pos, module, node_format):
        cell_node_name = node_format.format(*pos)
        sampling_param = None
        if self.sampling_param_generator is not None:
            sampling_param = self.sampling_param_generator(cell_node_name)

        self.graph.add_node(cell_node_name,
                            module=len(self.blocks),
                            module_params=module.params,
                            sampling_param=len(self.blocks),
                            structure_fixed=module.structure_fixed,
                            pos=NASDrawer.get_draw_pos(pos=pos))
    
        if sampling_param is not None:
            self.sampling_parameters.append(sampling_param)
    
        # 添加到层级结构中
        self.hierarchical[-1][-1].append(len(self.blocks))
        
        self.blocks.append(module)
        return cell_node_name

    def add_aggregation(self, pos, module, node_format):
        agg_node_name = node_format.format(*pos)
        sampling_param = None
        if self.sampling_param_generator is not None:
            sampling_param = self.sampling_param_generator(agg_node_name)

        self.graph.add_node(agg_node_name,
                            module=len(self.blocks),
                            module_params=module.params,
                            sampling_param=len(self.blocks),
                            structure_fixed=module.structure_fixed,
                            pos=NASDrawer.get_draw_pos(pos=pos))
    
        if sampling_param is not None:
            self.sampling_parameters.append(sampling_param)
    
        self.blocks.append(module)
        return agg_node_name

    def add_transformer(self, source, dest, module, src_node_format, des_node_format, transform_format):
        src_l, src_s = source
        dst_l, dst_s = dest
    
        trans_name = transform_format.format(src_l, src_s, dst_l, dst_s)
        source_name = src_node_format.format(src_l, src_s)
        dest_name = des_node_format.format(dst_l, dst_s)
    
        sampling_param = None
        if self.sampling_param_generator is not None:
            sampling_param = self.sampling_param_generator(trans_name)
    
        self.graph.add_node(trans_name,
                            module=len(self.blocks),
                            module_params=module.params,
                            sampling_param=len(self.blocks),
                            structure_fixed=module.structure_fixed,
                            pos=NASDrawer.get_draw_pos(source=source, dest=dest))
        self.graph.add_edge(source_name, trans_name, width_node=trans_name)
        self.graph.add_edge(trans_name, dest_name, width_node=trans_name)
    
        if sampling_param is not None:
            self.sampling_parameters.append(sampling_param)
    
        # 添加到层级结构中
        if transform_format.startswith("T"):
            self.hierarchical[-1][-1].append(len(self.blocks))
    
        self.blocks.append(module)
        return trans_name

    def mobilenetv2(self, head, scale=1.0):
        cfgs = [
            # t, c, n, s, r
            [1, int(16*scale), 1, 1, 1],
            [6, int(24*scale), 2, 2, 1],
            [6, int(32*scale), 3, 2, 1],
            [6, int(64*scale), 4, 2, 1],
            [6, int(96*scale), 3, 1, 1],
            [6, int(160*scale), 3, 1, 2],
            [6, int(320*scale), 1, 1, 1],
        ]
        input_channel = head.params['out_chan']
        width_mult = 1.0
        modules = []
        for t, c, n, s, r in cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                modules.append(InvertedResidualBlockWithSEHS(in_chan=input_channel,
                                                             out_chan=output_channel,
                                                             expansion=t,
                                                             kernel_size=3,
                                                             reduction=True if (i == 0) and (s == 2) else False,
                                                             se=False,
                                                             hs=False,
                                                             dilation=r))
                input_channel = output_channel

        self.in_node, _ = super(PKMixArc, self).generate(head, None, modules)
        return modules

    def build_backbone(self, head):
        if self.backbone.lower().startswith('mobilenetv2'):
            if '0.5' in self.backbone:
                return self.mobilenetv2(head, 0.5)
            elif '1.0' in self.backbone:
                return self.mobilenetv2(head, 1.0)

        return None

    def generate(self, head, tail, callback=None):
        # 1.step encoder (backbone)
        modules = self.build_backbone(head)

        # 2.step decoder (需要根据需要指定索引)
        backbone_export_endpoints = []
        for endpoint_index in self.decoder_input_endpoints:
            backbone_export_endpoints.append(self.names[endpoint_index])

        backbone_export_channels = []
        for endpoint_index in self.decoder_input_endpoints:
            backbone_export_channels.append(modules[endpoint_index].params['out_chan'])

        backbone_export_strides = self.decoder_input_strides
        pos_offset = self.offset
        
        # dense decoder
        decoder_channels = self.decoder_depth

        # aspp
        if self.aspp_cell_cls is not None:
            # aspp cell (接在backbone最后节点输出)
            self.add_fixed((0, pos_offset),
                           self.aspp_cell_cls(backbone_export_channels[-1], decoder_channels))
            self.graph.add_edge(backbone_export_endpoints[-1],
                                SuperNetwork._FIXED_NODE_FORMAT.format(0, pos_offset),
                                width_node=SuperNetwork._FIXED_NODE_FORMAT.format(0, pos_offset))
            backbone_export_endpoints[-1] = SuperNetwork._FIXED_NODE_FORMAT.format(0, pos_offset)
            backbone_export_channels[-1] = decoder_channels

            pos_offset += 1

        last_endpoints = backbone_export_endpoints
        last_channels = backbone_export_channels

        # dense decoder
        decoder_levels = len(backbone_export_endpoints)
        cells_pos = None
        for n in range(decoder_levels, 0, -1):
            cells_pos = [-1 for _ in range(n)]
            aggregation_pos = [-1 for _ in range(n)]
            
            for m in range(n-1, -1, -1):
                if self.decoder_allow_skip:
                    if n == decoder_levels:
                        self.add_cell((0, pos_offset),
                                      Skip(last_channels[m], last_channels[m]),
                                      SuperNetwork._CELL_NODE_FORMAT)

                        self.graph.add_edge(last_endpoints[m],
                                            SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset),
                                            width_node=SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset))
                        last_endpoints[m] = SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset)
                        pos_offset += 1

                for replica_i in range(self.decoder_replica):
                    upper_channels = last_channels[m] if replica_i == 0 else decoder_channels
                    upper_node = last_endpoints[m] if replica_i == 0 else SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset-1)
                    self.add_cell((0, pos_offset),
                                  self.cell_cls(upper_channels,
                                                decoder_channels,
                                                reduction=False),
                                  SuperNetwork._CELL_NODE_FORMAT)

                    self.graph.add_edge(upper_node,
                                        SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset),
                                        width_node=SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset))

                    cells_pos[m] = pos_offset
                    pos_offset += 1
                
                if m < n-1:
                    self.add_aggregation((0, pos_offset),
                                         self.aggregation_cls(),
                                         SuperNetwork._AGGREGATION_NODE_FORMAT)
                    
                    self.graph.add_edge(SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset-1),
                                        SuperNetwork._AGGREGATION_NODE_FORMAT.format(0, pos_offset),
                                        width_node=SuperNetwork._AGGREGATION_NODE_FORMAT.format(0, pos_offset))

                    aggregation_pos[m] = pos_offset
                    pos_offset += 1
                
            for m_1 in range(n-1, 0, -1):
                for m_2 in range(m_1 - 1, -1, -1):
                    scale_factor = backbone_export_strides[m_1]/backbone_export_strides[m_2]

                    # resize 模块为不可学习模块
                    self.add_fixed((0, pos_offset),
                                   ResizedBlock(decoder_channels, -1, scale_factor=scale_factor))
                    
                    self.graph.add_edge(SuperNetwork._CELL_NODE_FORMAT.format(0, cells_pos[m_1]),
                                        SuperNetwork._FIXED_NODE_FORMAT.format(0, pos_offset),
                                        width_node=SuperNetwork._FIXED_NODE_FORMAT.format(0, pos_offset))
                    
                    self.graph.add_edge(SuperNetwork._FIXED_NODE_FORMAT.format(0, pos_offset),
                                        SuperNetwork._AGGREGATION_NODE_FORMAT.format(0, aggregation_pos[m_2]),
                                        width_node=SuperNetwork._AGGREGATION_NODE_FORMAT.format(0, aggregation_pos[m_2]))
                    pos_offset += 1
                    
            last_channels = [decoder_channels for _ in range(n-1)]
            last_endpoints = [SuperNetwork._AGGREGATION_NODE_FORMAT.format(0, aggregation_pos[p]) for p in range(n-1)]

        output = SuperNetwork._CELL_NODE_FORMAT.format(0, cells_pos[0])
        if callback is not None:
            output, pos_offset = callback(output, pos_offset)

        # link output
        out_name = SuperNetwork._OUTPUT_NODE_FORMAT.format(0, pos_offset)
        self.graph.add_node(out_name,
                            module=len(self.blocks),
                            module_params=tail.params,
                            sampling_param=len(self.blocks),
                            structure_fixed=tail.structure_fixed,
                            pos=NASDrawer.get_draw_pos(pos=(0, pos_offset)),
                            sampled=1)
        self.blocks.append(tail)

        self.graph.add_edge(output,
                            SuperNetwork._OUTPUT_NODE_FORMAT.format(0, pos_offset),
                            width_node=SuperNetwork._OUTPUT_NODE_FORMAT.format(0, pos_offset))

        self.out_node = out_name
        return self.in_node, self.out_node