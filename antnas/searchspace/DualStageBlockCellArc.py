# -*- coding: UTF-8 -*-
# @Time    : 2020-03-31 12:42
# @File    : StageBlockCellArc.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antnas.networks.SuperNetwork import *
from antnas.utils.drawers.NASDrawer import NASDrawer
import torch
import torch.nn as nn
from antnas.searchspace.Arc import *


'''
StageBlockCell Architecture
stage
    block
        cell
        cell
    block
        cell
stage
    block
        cell
        cell
'''


class DualStageBlockCellArc(Arc):
    def __init__(self,
                 cell_cls,
                 reduction_cell_cls,
                 aggregation_cls,
                 transformer_cls,
                 graph,
                 sampling_param_generator=None,
                 cross_interval=1):
        super(DualStageBlockCellArc, self).__init__(graph)
        self.sampling_param_generator = sampling_param_generator
        self.sampling_parameters = nn.ParameterList()

        self.cell_cls = cell_cls
        self.reduction_cell_cls = reduction_cell_cls
        self.transformer_cls = transformer_cls
        self.aggregation_cls = aggregation_cls
        self.hierarchical = []
        self.cross_interval = cross_interval

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
        
        # 添加到层级结构中
        # if not (node_format.startswith("I") or node_format.startswith("O")):
        #     self.hierarchical[-1][-1].append(len(self.blocks))
        #
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
        self.graph.add_edge(source_name, trans_name,  width_node=trans_name)
        self.graph.add_edge(trans_name, dest_name,  width_node=trans_name)
        
        if sampling_param is not None:
            self.sampling_parameters.append(sampling_param)

        # 添加到层级结构中
        if transform_format.startswith("T"):
            self.hierarchical[-1][-1].append(len(self.blocks))
        
        self.blocks.append(module)
        return trans_name

    def add_block(self, cells, channles, pre_block_channels, reduction=False):
        pos_offset = self.offset
        for cell_i in range(cells):
            # Add
            # branch 1
            self.add_aggregation((0, pos_offset + cell_i * 4), self.aggregation_cls(),
                                 SuperNetwork._AGGREGATION_NODE_FORMAT)

            # branch 2
            self.add_aggregation((3, pos_offset + cell_i * 4+2), self.aggregation_cls(),
                                 SuperNetwork._AGGREGATION_NODE_FORMAT)

            # Cell
            if reduction and cell_i == 0:
                # branch 1
                self.add_cell((0, pos_offset + cell_i * 4 + 1),
                              self.reduction_cell_cls(pre_block_channels if cell_i == 0 else channles, channles),
                              SuperNetwork._CELL_NODE_FORMAT)

                # branch 2
                self.add_cell((3, pos_offset + cell_i * 4 + 3),
                              self.reduction_cell_cls(pre_block_channels if cell_i == 0 else channles, channles),
                              SuperNetwork._CELL_NODE_FORMAT)
            else:
                in_channels = pre_block_channels if cell_i == 0 else channles
                if cell_i >= self.cross_interval and self.aggregation_cls == ConcatBlock:
                    in_channels = in_channels * 2

                # branch 1
                self.add_cell((0, pos_offset + cell_i * 4 + 1),
                              self.cell_cls(in_channels,
                                            channles,
                                            reduction=False),
                              SuperNetwork._CELL_NODE_FORMAT)

                # branch 2
                self.add_cell((3, pos_offset + cell_i * 4 + 3),
                              self.cell_cls(in_channels,
                                            channles,
                                            reduction=False),
                              SuperNetwork._CELL_NODE_FORMAT)

            # 固定连接 (branch 1)
            self.graph.add_edge(SuperNetwork._AGGREGATION_NODE_FORMAT.format(0, pos_offset + cell_i * 4),
                                SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset + cell_i * 4 + 1),
                                width_node=SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset + cell_i * 4 + 1))

            # 固定连接（branch 2）
            self.graph.add_edge(SuperNetwork._AGGREGATION_NODE_FORMAT.format(3, pos_offset + cell_i * 4+2),
                                SuperNetwork._CELL_NODE_FORMAT.format(3, pos_offset + cell_i * 4 + 3),
                                width_node=SuperNetwork._CELL_NODE_FORMAT.format(3, pos_offset + cell_i * 4 + 3))

            if cell_i > 0:
                # 固定连接 (branch 1)
                self.graph.add_edge(SuperNetwork._CELL_NODE_FORMAT.format(0, pos_offset + (cell_i-1) * 4 + 1),
                                    SuperNetwork._AGGREGATION_NODE_FORMAT.format(0, pos_offset + cell_i * 4),
                                    width_node=SuperNetwork._AGGREGATION_NODE_FORMAT.format(0, pos_offset + cell_i * 4))

                # 固定连接 (branch 2)
                self.graph.add_edge(SuperNetwork._CELL_NODE_FORMAT.format(3, pos_offset + (cell_i-1) * 4 + 3),
                                    SuperNetwork._AGGREGATION_NODE_FORMAT.format(3, pos_offset + cell_i * 4 + 2),
                                    width_node=SuperNetwork._AGGREGATION_NODE_FORMAT.format(3, pos_offset + cell_i * 4 + 2))

            # 双branch交叉连接
            if cell_i >= self.cross_interval:
                # branch 1(cell_i-self.cross_interval) -> branch 2(cell_i)
                # skip
                self.add_transformer((0, pos_offset + (cell_i-self.cross_interval) * 4 + 1),
                                     (3, pos_offset + cell_i * 4 + 2),
                                     Skip(channles, channles, False),
                                     SuperNetwork._CELL_NODE_FORMAT,
                                     SuperNetwork._AGGREGATION_NODE_FORMAT,
                                     SuperNetwork._TRANSFORMATION_FORMAT)

                # branch 2(cell_i-self.cross_interval) -> branch 1(cell_i)
                # skip
                self.add_transformer((3, pos_offset + (cell_i-self.cross_interval) * 4 + 3),
                                     (0, pos_offset + cell_i * 4),
                                     Skip(channles, channles, False),
                                     SuperNetwork._CELL_NODE_FORMAT,
                                     SuperNetwork._AGGREGATION_NODE_FORMAT,
                                     SuperNetwork._TRANSFORMATION_FORMAT)

        self.offset += cells * 4
        return self.offset

    def add_stage(self, block_num, cells_per_block, channles_per_block, pre_stage_channels, is_first_stage):
        stage_offset = self.offset
        offset_per_block = []
        for block_i in range(block_num):
            self.hierarchical[-1].append([])
            offset_per_block.append(stage_offset)

            if block_i == 0:
                stage_offset = \
                    self.add_block(cells_per_block[block_i],
                                   channles_per_block[block_i],
                                   pre_stage_channels,
                                   True if (block_i == 0 and not is_first_stage) else False)
            else:
                stage_offset = \
                    self.add_block(cells_per_block[block_i],
                                   channles_per_block[block_i],
                                   channles_per_block[block_i-1],
                                   False)

            if block_i > 0:
                # branch 1(connect between blocks)
                pre_block_i = block_i - 1
                self.graph.add_edge(SuperNetwork._CELL_NODE_FORMAT.format(0, offset_per_block[pre_block_i] + cells_per_block[pre_block_i] * 4 - 3),
                                    SuperNetwork._AGGREGATION_NODE_FORMAT.format(0, offset_per_block[block_i]),
                                    width_node=SuperNetwork._AGGREGATION_NODE_FORMAT.format(0, offset_per_block[block_i]))

                # branch 2(connect between blocks)
                self.graph.add_edge(SuperNetwork._CELL_NODE_FORMAT.format(3,offset_per_block[pre_block_i] + cells_per_block[pre_block_i] * 4 - 1),
                                    SuperNetwork._AGGREGATION_NODE_FORMAT.format(3, offset_per_block[block_i]+2),
                                    width_node=SuperNetwork._AGGREGATION_NODE_FORMAT.format(3, offset_per_block[block_i]+2))

        self.offset = stage_offset
        return self.offset

    def generate(self, head, tail, blocks, cells, channels):
        in_name = self.add_aggregation((0, 0), module=head, node_format=SuperNetwork._INPUT_NODE_FORMAT)
        self.offset += 1
        
        offset_per_stage = []
        for stage_i in range(len(blocks)):
            self.hierarchical.append([])
            offset_per_stage.append(self.offset)
            self.add_stage(blocks[stage_i],
                           cells[stage_i],
                           channels[stage_i],
                           head.params['out_chan'] if stage_i == 0 else channels[stage_i - 1][-1],
                           stage_i == 0)
                        
            if stage_i > 0:
                # channel 1 (connect between stages)
                self.graph.add_edge(SuperNetwork._CELL_NODE_FORMAT.format(*(0, offset_per_stage[stage_i-1]+sum(cells[stage_i-1])*4-3)),
                                    SuperNetwork._AGGREGATION_NODE_FORMAT.format(*(0, offset_per_stage[stage_i])),
                                    width_node=SuperNetwork._AGGREGATION_NODE_FORMAT.format(*(0, offset_per_stage[stage_i])))

                # channel 2 (connect between stages)
                self.graph.add_edge(SuperNetwork._CELL_NODE_FORMAT.format(*(3, offset_per_stage[stage_i-1]+sum(cells[stage_i-1])*4-1)),
                                    SuperNetwork._AGGREGATION_NODE_FORMAT.format(*(3, offset_per_stage[stage_i]+2)),
                                    width_node=SuperNetwork._AGGREGATION_NODE_FORMAT.format(*(3, offset_per_stage[stage_i]+2)))

            if stage_i == 0:
                # link head to channel 1 and channel 2
                self.graph.add_edge(SuperNetwork._INPUT_NODE_FORMAT.format(0, 0),
                                    SuperNetwork._AGGREGATION_NODE_FORMAT.format(0, 1),
                                    width_node=SuperNetwork._AGGREGATION_NODE_FORMAT.format(0, 1))

                self.graph.add_edge(SuperNetwork._INPUT_NODE_FORMAT.format(0, 0),
                                    SuperNetwork._AGGREGATION_NODE_FORMAT.format(3, 3),
                                    width_node=SuperNetwork._AGGREGATION_NODE_FORMAT.format(3, 3))

        # link channel 1 and channel 2
        # (channel 1 + channel 2)
        self.add_aggregation((0, offset_per_stage[-1] + sum(cells[-1]) * 4), self.aggregation_cls(),
                             SuperNetwork._AGGREGATION_NODE_FORMAT)

        self.graph.add_edge(SuperNetwork._CELL_NODE_FORMAT.format(*(0, offset_per_stage[-1] + sum(cells[-1]) * 4 - 3)),
                            SuperNetwork._AGGREGATION_NODE_FORMAT.format(*(0, offset_per_stage[-1] + sum(cells[-1]) * 4)),
                            width_node=SuperNetwork._AGGREGATION_NODE_FORMAT.format(*(0, offset_per_stage[-1] + sum(cells[-1]) * 4)))

        self.graph.add_edge(SuperNetwork._CELL_NODE_FORMAT.format(*(3, offset_per_stage[-1] + sum(cells[-1]) * 4 - 1)),
                            SuperNetwork._AGGREGATION_NODE_FORMAT.format(*(0, offset_per_stage[-1] + sum(cells[-1]) * 4)),
                            width_node=SuperNetwork._AGGREGATION_NODE_FORMAT.format(*(0, offset_per_stage[-1] + sum(cells[-1]) * 4)))

        # link to tail
        out_name = SuperNetwork._OUTPUT_NODE_FORMAT.format(*(0, offset_per_stage[-1] + sum(cells[-1]) * 4 + 1))

        sampling_param = None
        if self.sampling_param_generator is not None:
            sampling_param = self.sampling_param_generator(out_name)
        self.graph.add_node(out_name,
                            module=len(self.blocks),
                            module_params=tail.params,
                            sampling_param=len(self.blocks),
                            structure_fixed=tail.structure_fixed,
                            pos=NASDrawer.get_draw_pos(pos=(0, offset_per_stage[-1] + sum(cells[-1]) * 4 + 1)))
        self.graph.add_edge(
            SuperNetwork._AGGREGATION_NODE_FORMAT.format(*(0, offset_per_stage[-1] + sum(cells[-1]) * 4)),
            out_name,
            width_node=out_name)

        if sampling_param is not None:
            self.sampling_parameters.append(sampling_param)
        self.blocks.append(tail)

        self.in_node = in_name
        self.out_node = out_name
        return in_name, out_name