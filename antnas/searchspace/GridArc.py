# -*- coding: UTF-8 -*-
# @Time    : 2020/9/20 1:42 下午
# @File    : GridArc.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antnas.networks.SuperNetwork import *
from antnas.utils.drawers.NASDrawer import NASDrawer
import torch
import torch.nn as nn
from antnas.searchspace.Arc import *


class GridArc(Arc):
    def __init__(self,
                 graph,
                 grid_h,
                 grid_w,
                 cell_cls,
                 skip_transformer_cls,
                 upsample_transformer_cls,
                 downsample_transformer_cls,
                 add_aggregate_cls,
                 num_blocks=4,
                 num_channels=[48, 96, 192, 384]):
        super(GridArc, self).__init__(graph)
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.cell_cls = cell_cls
        self.skip_transformer_cls = skip_transformer_cls
        self.upsample_transformer_cls = upsample_transformer_cls            # 1x1 conv+bn + resize
        self.downsample_transformer_cls = downsample_transformer_cls        # 3x3 stride conv
        self.aggregation_cls = add_aggregate_cls                            # add + relu
        self.concat_aggregation_cls = ConcatBlock                           # concat
        self.num_blocks = num_blocks
        self.num_channels = num_channels  # len(self.num_channels) == self.grid_h
        self.offset = 0

    def add_cell(self, pos, module, node_format):
        cell_node_name = node_format.format(*pos)

        self.graph.add_node(cell_node_name,
                            module=len(self.blocks),
                            module_params=module.params,
                            sampling_param=len(self.blocks),
                            structure_fixed=module.structure_fixed,
                            pos=NASDrawer.get_draw_pos(pos=pos))

        self.blocks.append(module)
        return cell_node_name

    def add_aggregation(self, pos, module, node_format):
        agg_node_name = node_format.format(*pos)
        self.graph.add_node(agg_node_name,
                            module=len(self.blocks),
                            module_params=module.params,
                            sampling_param=len(self.blocks),
                            structure_fixed=module.structure_fixed,
                            pos=NASDrawer.get_draw_pos(pos=pos))

        self.blocks.append(module)
        return agg_node_name

    def add_transformer(self, source, dest, module, src_node_format, des_node_format, transform_format):
        src_l, src_s = source
        dst_l, dst_s = dest

        trans_name = transform_format.format(src_l, src_s, dst_l, dst_s)
        source_name = src_node_format.format(src_l, src_s)
        dest_name = des_node_format.format(dst_l, dst_s)

        self.graph.add_node(trans_name,
                            module=len(self.blocks),
                            module_params=module.params,
                            sampling_param=len(self.blocks),
                            structure_fixed=module.structure_fixed,
                            pos=NASDrawer.get_draw_pos(source=source, dest=dest))
        self.graph.add_edge(source_name, trans_name, width_node=trans_name)
        self.graph.add_edge(trans_name, dest_name, width_node=trans_name)

        self.blocks.append(module)
        return trans_name

    def add_module(self, grid_i, grid_j, entry_channels=None):
        # 获得此module的channels
        out_channels = self.num_channels[grid_i]

        # Add
        NODE_FORMAT = SuperNetwork._BINARY_NODE_FORMAT
        if grid_i == 0 and grid_j == 0:
            NODE_FORMAT = SuperNetwork._AGGREGATION_NODE_FORMAT
        self.add_aggregation((self.offset+grid_i, grid_j * (self.num_blocks + 1)),
                             self.aggregation_cls(),
                             NODE_FORMAT)

        # Cells
        for block_i in range(self.num_blocks):
            if block_i == 0 and entry_channels is not None:
                in_channels = entry_channels
            else:
                in_channels = out_channels

            self.add_cell((self.offset+grid_i, grid_j * (self.num_blocks + 1) + block_i + 1),
                          self.cell_cls(in_channels, out_channels),
                          SuperNetwork._CELL_NODE_FORMAT)

            if block_i == 0:
                self.graph.add_edge(
                    NODE_FORMAT.format(self.offset + grid_i, grid_j * (self.num_blocks + 1)),
                    SuperNetwork._CELL_NODE_FORMAT.format(self.offset + grid_i,
                                                          grid_j * (self.num_blocks + 1) + block_i + 1),
                    width_node=SuperNetwork._CELL_NODE_FORMAT.format(self.offset + grid_i,
                                                                     grid_j * (self.num_blocks + 1) + block_i + 1))
            else:
                self.graph.add_edge(
                    SuperNetwork._CELL_NODE_FORMAT.format(self.offset+grid_i, grid_j * (self.num_blocks + 1) + block_i),
                    SuperNetwork._CELL_NODE_FORMAT.format(self.offset+grid_i, grid_j * (self.num_blocks + 1) + block_i + 1),
                    width_node=SuperNetwork._CELL_NODE_FORMAT.format(self.offset+grid_i, grid_j * (self.num_blocks + 1) + block_i + 1))

    def generate(self, head, tail):
        # 设置行偏移
        entry_channels = 0
        if type(head) == list:
            self.offset = len(head)
            entry_channels = head[-1].params['out_chan']
        else:
            self.offset = 1
            entry_channels = head.params['out_chan']

        # 1.step 添加网格节点模块
        for grid_i in range(self.grid_h):
            for grid_j in range(self.grid_w):
                if grid_i <= grid_j:
                    if grid_i == 0 and grid_j == 0:
                        self.add_module(grid_i, grid_j, entry_channels)
                    else:
                        self.add_module(grid_i, grid_j)

        # 2.1.step 添加网格节点间的链接
        for grid_i in range(self.grid_h):
            for grid_j in range(self.grid_w):
                if grid_i > grid_j or grid_j == 0:
                    continue

                # 链接(:,grid_jj)到当前(grid_i,grid_j)
                fusion_num = min(self.grid_h, grid_j)
                grid_jj = grid_j - 1
                for grid_ii in range(fusion_num):
                    if grid_ii == grid_i:
                        self.add_transformer((self.offset+grid_ii, grid_jj * (self.num_blocks + 1) + self.num_blocks),
                                             (self.offset+grid_i, grid_j * (self.num_blocks + 1)),
                                             self.skip_transformer_cls(self.num_channels[grid_ii],
                                                                       self.num_channels[grid_i]),
                                             SuperNetwork._CELL_NODE_FORMAT,
                                             SuperNetwork._BINARY_NODE_FORMAT,
                                             SuperNetwork._TRANSFORMATION_FORMAT)
                    elif grid_ii > grid_i:
                        # upsample transformer
                        self.add_transformer((self.offset+grid_ii, grid_jj * (self.num_blocks + 1) + self.num_blocks),
                                             (self.offset+grid_i, grid_j * (self.num_blocks + 1)),
                                             self.upsample_transformer_cls(self.num_channels[grid_ii],
                                                                           self.num_channels[grid_i],
                                                                           np.power(2, grid_ii-grid_i)),
                                             SuperNetwork._CELL_NODE_FORMAT,
                                             SuperNetwork._BINARY_NODE_FORMAT,
                                             SuperNetwork._TRANSFORMATION_FORMAT)
                    else:
                        # downsample transformer
                        self.add_transformer((self.offset+grid_ii, grid_jj * (self.num_blocks + 1) + self.num_blocks),
                                             (self.offset+grid_i, grid_j * (self.num_blocks + 1)),
                                             self.downsample_transformer_cls(self.num_channels[grid_ii],
                                                                             self.num_channels[grid_i],
                                                                             np.power(2, grid_i-grid_ii)),
                                             SuperNetwork._CELL_NODE_FORMAT,
                                             SuperNetwork._BINARY_NODE_FORMAT,
                                             SuperNetwork._TRANSFORMATION_FORMAT)

        # 2.2.step fuse output
        for grid_i in range(self.grid_h):
            self.add_aggregation((self.offset+grid_i, self.grid_w*(self.num_blocks + 1)),
                                 self.aggregation_cls(),
                                 SuperNetwork._BINARY_NODE_FORMAT)

            for grid_ii in range(self.grid_h):
                if grid_ii == grid_i:
                    self.add_transformer((self.offset+grid_ii, (self.grid_w - 1) * (self.num_blocks + 1) + self.num_blocks),
                                         (self.offset+grid_i, self.grid_w * (self.num_blocks + 1)),
                                         self.skip_transformer_cls(self.num_channels[grid_ii],
                                                                   self.num_channels[grid_i]),
                                         SuperNetwork._CELL_NODE_FORMAT,
                                         SuperNetwork._BINARY_NODE_FORMAT,
                                         SuperNetwork._TRANSFORMATION_FORMAT)
                elif grid_ii > grid_i:
                    # upsample transformer
                    self.add_transformer((self.offset+grid_ii, (self.grid_w - 1) * (self.num_blocks + 1) + self.num_blocks),
                                         (self.offset+grid_i, self.grid_w * (self.num_blocks + 1)),
                                         self.upsample_transformer_cls(self.num_channels[grid_ii],
                                                                       self.num_channels[grid_i],
                                                                       np.power(2, grid_ii - grid_i)),
                                         SuperNetwork._CELL_NODE_FORMAT,
                                         SuperNetwork._BINARY_NODE_FORMAT,
                                         SuperNetwork._TRANSFORMATION_FORMAT)
                else:
                    # downsample transformer
                    self.add_transformer((self.offset+grid_ii, (self.grid_w - 1) * (self.num_blocks + 1) + self.num_blocks),
                                         (self.offset+grid_i, self.grid_w * (self.num_blocks + 1)),
                                         self.downsample_transformer_cls(self.num_channels[grid_ii],
                                                                         self.num_channels[grid_i],
                                                                         np.power(2, grid_i - grid_ii)),
                                         SuperNetwork._CELL_NODE_FORMAT,
                                         SuperNetwork._BINARY_NODE_FORMAT,
                                         SuperNetwork._TRANSFORMATION_FORMAT)

        # 3.step 连接头结点
        in_name = ''
        if type(head) == list:
            for head_i in range(len(head)):
                name = self.add_aggregation((head_i, 0), module=head[head_i], node_format=SuperNetwork._INPUT_NODE_FORMAT)
                if head_i == 0:
                    in_name = name

                if head_i > 0:
                    self.graph.add_edge(SuperNetwork._INPUT_NODE_FORMAT.format(head_i - 1, 0),
                                        SuperNetwork._INPUT_NODE_FORMAT.format(head_i, 0),
                                        width_node=SuperNetwork._INPUT_NODE_FORMAT.format(head_i, 0))

                    if head_i == len(head) - 1:
                        self.graph.add_edge(SuperNetwork._INPUT_NODE_FORMAT.format(head_i, 0),
                                            SuperNetwork._AGGREGATION_NODE_FORMAT.format(self.offset, 0),
                                            width_node=SuperNetwork._AGGREGATION_NODE_FORMAT.format(self.offset, 0))
        else:
            in_name = self.add_aggregation((0, 0), module=head, node_format=SuperNetwork._INPUT_NODE_FORMAT)
            self.graph.add_edge(SuperNetwork._INPUT_NODE_FORMAT.format(0, 0),
                                SuperNetwork._AGGREGATION_NODE_FORMAT.format(self.offset, 0),
                                width_node=SuperNetwork._AGGREGATION_NODE_FORMAT.format(self.offset, 0))

        # 4.step 连接尾节点
        self.add_aggregation((0, (self.grid_w+1) * (self.num_blocks + 1)),
                             self.concat_aggregation_cls(),
                             SuperNetwork._AGGREGATION_NODE_FORMAT)

        for grid_ii in range(self.grid_h):
            if grid_ii == 0:
                self.add_transformer(
                    (self.offset + grid_ii, self.grid_w * (self.num_blocks + 1)),
                    (0, (self.grid_w+1) * (self.num_blocks + 1)),
                    self.skip_transformer_cls(self.num_channels[grid_ii], self.num_channels[0]),
                    SuperNetwork._BINARY_NODE_FORMAT,
                    SuperNetwork._AGGREGATION_NODE_FORMAT,
                    SuperNetwork._TRANSFORMATION_FORMAT)
            elif grid_ii > 0:
                # upsample transformer
                self.add_transformer(
                    (self.offset + grid_ii, self.grid_w * (self.num_blocks + 1)),
                    (0, (self.grid_w+1) * (self.num_blocks + 1)),
                    self.upsample_transformer_cls(self.num_channels[grid_ii],
                                                  self.num_channels[0],
                                                  np.power(2, grid_ii - 0)),
                    SuperNetwork._BINARY_NODE_FORMAT,
                    SuperNetwork._AGGREGATION_NODE_FORMAT,
                    SuperNetwork._TRANSFORMATION_FORMAT)

        out_name = SuperNetwork._OUTPUT_NODE_FORMAT.format(*(0, (self.grid_w+2) * (self.num_blocks + 1)))
        self.graph.add_node(out_name,
                            module=len(self.blocks),
                            module_params=tail.params,
                            sampling_param=len(self.blocks),
                            structure_fixed=tail.structure_fixed,
                            pos=NASDrawer.get_draw_pos(pos=(0, (self.grid_w+2) * (self.num_blocks + 1))))
        self.graph.add_edge(
            SuperNetwork._AGGREGATION_NODE_FORMAT.format(*(0, (self.grid_w+1) * (self.num_blocks + 1))),
            out_name,
            width_node=out_name)
        self.blocks.append(tail)

        return in_name, out_name