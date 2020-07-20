import numpy as np
from antnas.utils.drawers.Drawer import Drawer
import networkx as nx
import matplotlib.pyplot as plt


class NASDrawer(Drawer):
    NODE_SIZE = 100
    EDGE_WIDTH = 1
    
    @staticmethod
    def caculate_parabola_and_get_pos(pos1, pos2):
        p_x1, p_y1 = pos1
        p_x2, p_y2 = pos2
        
        vx = p_x2-p_x1 + 0.00000001
        vy = p_y2-p_y1 + 0.00000001
        
        length = np.sqrt(np.power(p_x1-p_x2, 2.0)+np.power(p_y1-p_y2, 2.0))
        offset_size = 0.3 * length
        
        p_xc, p_yc = (p_x1+p_x2)/2, (p_y1+p_y2)/2
        
        target_x = 0.5 + np.random.random() * 2
        target_y = p_yc + (np.random.random() * 2 - 1.0)
        
        return target_x, target_y
    
    @staticmethod
    def get_draw_pos(source=None, dest=None, pos=None):
        if pos is not None:
            return pos[0], pos[1]
        
        if source is not None and dest is not None:
            return NASDrawer.caculate_parabola_and_get_pos(source, dest)

    def draw(self,
             graph,
             param_list=None,
             weights=None,
             colormap=None):
        node_filter = lambda n: True  # todo Should be removed
        edge_filter = lambda e: True  # todo Should be removed

        if param_list is not None:
            def weighter(e):
                width_node = graph.node[graph.get_edge_data(*e)['width_node']]
                return param_list[width_node['sampling_param']].data[0]
        elif weights is None:
            def weighter(e):
                width_node = graph.node[graph.get_edge_data(*e)['width_node']]
                if 'structure_fixed' in width_node and width_node['structure_fixed']:
                    # 结构固定节点
                    return 1

                if 'structure_fixed' in width_node and not width_node['structure_fixed']:
                    # 结构不固定节点，根据当前采样值返回
                    if 'sampling_val' in width_node and width_node['sampling_val'] is not None:
                        return width_node['sampling_val']
                    
                    if 'sampled' in width_node and width_node['sampled'] is not None:
                        return width_node['sampled']
                    
                    return 1
                    
                return 1
        elif type(weights) is float:
            def weighter(_):
                return weights
        else:
            def weighter(e):
                return weights[graph.get_edge_data(*e)['width_node']]
        
        def samplinger(n):
            node = graph.node[n]
            if 'sampling_val' in node:
                return (int)(node['sampling_val'])
            elif 'sampled' in node:
                return (int)(node['sampled'])
            
            return 1
        
        def positioner(n):
            if 'pos' not in graph.node[n]:
                return None
            return graph.node[n]['pos']

        img = self._draw_net(graph,
                             nodefilter=node_filter,
                             samplinger=samplinger,
                             edgefilter=edge_filter,
                             positioner=positioner,
                             weighter=weighter,
                             colormap=colormap)
        
        #
        self.win = self.vis.svg(svgstr=img, win=self.win)


if __name__ == '__main__':
    G = nx.Graph()
    G.add_node(1)
    G.add_node('A')
    G.add_nodes_from([2, 3])
    G.add_edges_from([(1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (4, 8), (5, 8), (3, 7)])
    
    DD = [(index, 0) for index in range(len(G.nodes()))]
    
    pos = dict((n, DD[ni]) for ni,n in enumerate(G.nodes()))
    labels = dict((n, str(n)) for ni,n in enumerate(G.nodes()))
    
    nx.draw_networkx_nodes(G,pos=pos,node_color='red')
    nx.draw_networkx_edges(G,pos=pos,edge_color='blue',edge_cmap=plt.cm.YlGnBu)
    nx.draw_networkx_labels(G,pos=pos,labels=labels)
    # H = nx.path_graph(10)
    # G.add_nodes_from(H)

    # nx.draw(G, with_labels=True)
    # plt.show()
    # fig = plt.figure()
    # G = nx.Graph()
    # # tuple
    # G.add_edges_from([('A',2),('A',3),(2,4),(2,5),(3,6),(4,8),(5,8),(3,7)])
    #
    # nx.draw(G, with_labels=True, edge_color='b', node_color='g', node_size=1000)
    #

    plt.savefig('pic.png', bbox_inches='tight')
