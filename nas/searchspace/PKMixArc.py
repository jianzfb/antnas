# -*- coding: UTF-8 -*-
# @Time    : 2020-05-15 19:05
# @File    : PKMixArc.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from nas.searchspace.Arc import *
from nas.searchspace.PKAutoArc import *
from nas.component.NetworkBlock import *
from nas.component.NetworkBlock import _make_divisible


class PKMixArc(PKAutoArc):
    def __init__(self,
                 cell_cls,
                 aspp_cell_cls,
                 aggregation_cls,
                 transformer_cls,
                 graph,
                 sampling_param_generator=None):
        super(PKMixArc, self).__init__(graph)
        self.sampling_param_generator = sampling_param_generator
        self.sampling_parameters = nn.ParameterList()

        self.cell_cls = cell_cls
        self.aspp_cell_cls = aspp_cell_cls
        self.transformer_cls = transformer_cls
        self.aggregation_cls = aggregation_cls
        self.hierarchical = [[[]]]
    
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
    
    def generate(self, head, tail):
        # this is Example Code (for seg architecture)
        # mobilenet-v2 1.0
        # setting of inverted residual blocks
        # cfgs = [
        #     # t, c, n, s, r
        #     [1, 16, 1, 1, 1],
        #     [6, 24, 2, 2, 1],
        #     [6, 32, 3, 2, 1],
        #     [6, 64, 4, 2, 1],
        #     [6, 96, 3, 1, 1],
        #     [6, 160, 3, 2, 1],
        #     [6, 320, 1, 1, 1],
        # ]
        cfgs = [
            # t, c, n, s, r
            [1, 16, 1, 1, 1],
            [6, 24, 2, 2, 1],
            [6, 32, 3, 2, 1],
            [6, 64, 4, 2, 1],
            [6, 96, 3, 1, 1],
            [6, 160, 3, 1, 2],
            [6, 320, 1, 1, 1],
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

        # 1.step encoder (backbone)
        self.in_node, _ = super(PKMixArc, self).generate(head, None, modules)
        
        # 2.step decoder (需要根据需要指定索引)
        layer_1_extract = 0
        layer_2_extract = 2
        layer_3_extract = 5
        layer_4_extract = 16
        
        layer_1_endpoint = self.names[layer_1_extract]
        layer_2_endpoint = self.names[layer_2_extract]
        layer_3_endpoint = self.names[layer_3_extract]
        layer_4_endpoint = self.names[layer_4_extract]
        
        pos_offset = self.offset
        
        # dense decoder
        decoder_channels = 64
        endpoints = [layer_1_endpoint,
                     layer_2_endpoint,
                     layer_3_endpoint,
                     layer_4_endpoint]
        
        channels = [modules[layer_1_extract].params['out_chan'],
                    modules[layer_2_extract].params['out_chan'],
                    modules[layer_3_extract].params['out_chan'],
                    modules[layer_4_extract].params['out_chan']]

        # aspp cell
        self.add_cell((0, pos_offset),
                      self.aspp_cell_cls(channels[-1], decoder_channels),
                      SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset))
        self.graph.add_edge(layer_4_endpoint,
                            SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset),
                            width_node=SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset))
        endpoints[-1] = SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset)
        channels[-1] = decoder_channels
        
        pos_offset += 1

        last_endpoints = endpoints
        last_channels = channels
        cells_pos = None
        for n in range(4, 0, -1):
            cells_pos = [-1 for _ in range(n)]
            aggregation_pos = [-1 for _ in range(n)]
            
            for m in range(n-1, -1, -1):
                if n == 4:
                    self.add_cell((0, pos_offset),
                                  Skip(last_channels[m], last_channels[m]),
                                  SuperNetwork._CELL_NODE_FORMAT)
    
                    self.graph.add_edge(last_endpoints[m],
                                        SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset),
                                        width_node=SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset))
                    last_endpoints[m] = SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset)
                    pos_offset += 1
                
                self.add_cell((0, pos_offset),
                              self.cell_cls(last_channels[m],
                                            decoder_channels,
                                            reduction=False),
                              SuperNetwork._CELL_NODE_FORMAT)

                self.graph.add_edge(last_endpoints[m],
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
                    scale_factor = pow(2, m_1 - m_2)
                    self.add_cell((0, pos_offset),
                                  ResizedBlock(decoder_channels, -1, scale_factor=scale_factor),
                                  SuperNetwork._CELL_NODE_FORMAT)
                    
                    self.graph.add_edge(SuperNetwork._CELL_NODE_FORMAT.format(0, cells_pos[m_1]),
                                        SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset),
                                        width_node=SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset))
                    
                    self.graph.add_edge(SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset),
                                        SuperNetwork._AGGREGATION_NODE_FORMAT.format(0, aggregation_pos[m_2]),
                                        width_node=SuperNetwork._AGGREGATION_NODE_FORMAT.format(0, aggregation_pos[m_2]))
                    pos_offset += 1
                    
            last_channels = [decoder_channels for _ in range(n-1)]
            last_endpoints = [SuperNetwork._AGGREGATION_NODE_FORMAT.format(0, aggregation_pos[p]) for p in range(n-1)]
           
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

        self.graph.add_edge(SuperNetwork._CELL_NODE_FORMAT.format(0,cells_pos[0]),
                            SuperNetwork._OUTPUT_NODE_FORMAT.format(0, pos_offset),
                            width_node=SuperNetwork._OUTPUT_NODE_FORMAT.format(0, pos_offset))

        self.out_node = out_name
        return self.in_node, self.out_node