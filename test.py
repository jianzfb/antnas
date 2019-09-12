# @Time    : 2019-08-23 15:28
# @Author  : zhangchenming
import torch.nn.functional as F
import torch
import threading
import networkx as nx

# se_x = torch.randn([3, 2])
# # print(se_x)
# se_x = -se_x
# print(se_x)
# se_x = F.threshold(se_x, -1, -1)
# print(se_x)
# se_x = F.threshold(-se_x, 0, 0)
#
# print(se_x)

#
# class A(object):
#     def __init__(self):
#         self.obj = threading.local()
#
#     def hellorun(self):
#         for _ in range(10):
#             try:
#                 self.obj.x += 1
#             except:
#                 self.obj.x = 1
#
#         print('hello im %d'%self.obj.x)
#
# a = A()
#
# b = a.__new__(A)
# b.__dict__ = a.__dict__.copy()
#
#
# t1 = threading.Thread(target=a.hellorun)
# t2 = threading.Thread(target=b.hellorun)
# t1.start()
# t2.start()
#
# t1.join()
# t2.join()
#
# print(a.obj.x)
# print(b.obj.x)


graph = nx.read_gpickle("./nas_0.architecture")
travel = list(nx.topological_sort(graph))


layers_map = {}
i = 0
for node_name in travel:
    print()
    print(node_name)
    cur_node = graph.node[node_name]
    for pre_name in graph.predecessors(node_name):
        print(pre_name)