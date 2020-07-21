import io
import json
import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import antvis.client.mlogger as mlogger

logger = logging.getLogger(__name__)


class Drawer(object):
    __default_env = None

    def __init__(self):
        self.win = None
        # self.vis = mlogger.vis

    def draw_weights(self, graph, weights=None, vis_opts=None, vis_win=None, vis_env=None):
        node_filter = lambda n: True
        edge_filter = lambda e: True

        if weights is None:
            def weighter(e):
                width_node = graph.node[graph.get_edge_data(*e)['width_node']]
                if 'cost' in width_node.keys():
                    return width_node['cost']
                else:
                    return 1
        else:
            raise RuntimeError("draw_weights can't have weights param")

        def positioner(n):
            if 'pos' not in graph.node[n]:
                return None
            return graph.node[n]['pos']

        img = self._draw_net(graph,
                             nodefilter=node_filter,
                             edgefilter=edge_filter,
                             positioner=positioner,
                             weighter=weighter)

        env = vis_env if vis_env is not None else self.env
        win = vis_win if vis_win is not None else self.win
        if 'width' not in vis_opts:
            vis_opts['width'] = 600
        if 'height' not in vis_opts:
            vis_opts['height'] = 450
        
        # self.win = self.vis.svg(svgstr=img, win=win, opts=vis_opts, env=env)

    def _draw_net(self, graph,
                  filename=None,
                  show_fig=False,
                  normalize=False,
                  nodefilter=None,
                  samplinger=None,
                  edgefilter=None,
                  positioner=None,
                  weighter=None,
                  colormap=None):
        plt.close()
        plt.figure(figsize=(12, 12))
        
        nodes = None
        if nodefilter:
            nodes = [node for node in graph.nodes() if nodefilter(node)]
        
        nodes_value = None
        if samplinger:
            nodes_value = [samplinger(node) for node in graph.nodes() if nodefilter(node)]
        
        edges = None
        if edgefilter:
            edges = [edge for edge in graph.edges() if edgefilter(edge)]

        if positioner is None:
            pos = nx.spring_layout(graph)
        else:
            pos = dict((n, positioner(n)) for n in graph.nodes())
        
        pos_list = [positioner(n) for n in graph.nodes()]
        pos_x = [pos_list[index][0] for index in range(len(pos_list))]
        pos_min_x = np.min(pos_x)
        pos_max_x = np.max(pos_x)
        pos_y = [pos_list[index][1] for index in range(len(pos_list))]
        pos_min_y = np.min(pos_y)
        pos_max_y = np.max(pos_y)
        
        nodes_label = dict((n, str(n)) for n in graph.nodes())
        
        weights = 1.0
        if weighter is not None:
            weights = [weighter(e) for e in edges]

        weights = np.array(weights)
        w_min = weights.min()
        w_max = weights.max()
        if normalize and w_min != w_max:
            weights = np.log(weights + 1e-5)
            weights = (weights - w_min) * 1.0 / (w_max - w_min) + 2

        v_min = w_min - .1

        if colormap is None:
            colormap = plt.cm.YlGnBu
        
        # 节点类型
        # 1.输入节点 - I_{}_{}                       黄色 #FFFF00
        # 2.输出节点 - O_{}_{}                       黄色 #FFFF00
        # 3.Cell节点（可学习） - CELL_{}_{}           红橙黄绿青蓝紫 （#FF0000,#FF7D00,#FFFF00,#00FF00,#00FFFF, #0000FF,#FF00FF）
        # 4.连接节点（可学习）- T_{}_{}-{}_{}          红/绿色 #FF0000, #00FF00
        # 5.连接节点（不可学习）- L_{}_{}-{}_{}        绿色  #00FF00
        # 6.固定节点（不可学习）- FIXED_{}_{}          绿色  #00FF00
        
        cell_color = ['#FF0000', '#FF7D00', '#FFFF00', '#00FF00', '#00FFFF', '#0000FF', '#FF00FF']
        transfer_color = ['#FF0000', '#00FF00']
        
        node_color = []
        for node, value in zip(nodes, nodes_value):
            if node.startswith('I'):
                node_color.append('#FF6347')
            elif node.startswith('O'):
                node_color.append('#FF6347')
            elif node.startswith('CELL'):
                node_color.append(cell_color[value])
            elif node.startswith('T'):
                node_color.append(transfer_color[value])
            elif node.startswith('L') or node.startswith('FIXED'):
                node_color.append('#00FF00')
            else:
                node_color.append('#A9A9A9')
        
        nx.draw_networkx_nodes(graph,
                               nodelist=nodes,
                               pos=pos,
                               node_size=self.NODE_SIZE,
                               node_color=node_color)
        
        nx.draw_networkx_labels(graph,
                                pos=pos,
                                labels=nodes_label,
                                font_size=6)
        
        res = nx.draw_networkx_edges(graph,
                                     edgelist=edges,
                                     pos=pos,
                                     width=self.EDGE_WIDTH,
                                     arrows=False,
                                     edge_color=weights,
                                     edge_cmap=colormap,
                                     edge_vmin=v_min)
        
        plt.colorbar(res)
        plt.axis('off')
        plt.xlim(-1, 5)
        # plt.ylim(pos_min_y-5,pos_max_y + 5)
        #
        if show_fig:
            plt.show()
        if filename is not None:
            plt.savefig(filename, format='svg')

        img_data = io.StringIO()
        plt.savefig(img_data, format='svg')

        return img_data.getvalue()

    # def scatter(self, x, y, opts, vis_win):
    #     points = []
    #     labels = []
    #     legend = []
    #
    #     for i, (name, abs) in enumerate(x.items()):
    #         ord = 0 if y is None else y[name]
    #
    #         points.append((abs, ord))
    #         labels.append(i + 1)
    #         legend.append(name)
    #
    #     points = np.asarray(points)
    #
    #     vis_opts = dict(
    #         legend=legend,
    #         markersize=5,
    #     )
    #     vis_opts.update(opts)
    #     self.vis.scatter(points, labels, win=vis_win, opts=vis_opts, env=self.env)
