import logging
import torch
from nas.implem.EdgeCostEvaluator import EdgeCostEvaluator
from nas.interfaces.NetworkBlock import NetworkBlock

logger = logging.getLogger(__name__)


class ComputationalCostEvaluator(EdgeCostEvaluator):
    def init_costs(self, model, main_cost):
        with torch.no_grad():
            input_var = (torch.ones(1, *model.input_size),)
            graph = model.net

            self.costs = torch.Tensor(graph.number_of_nodes(), NetworkBlock.state_num)

            graph.node[model.in_node]['input'] = [*input_var]
            # graph.node[model.in_node]['input'] = input_var

            for node in model.traversal_order:
                cur_node = graph.node[node]
                input_var = model.format_input(cur_node['input'])

                print(node)
                out = cur_node['module'](input_var)

                if isinstance(cur_node['module'], NetworkBlock):
                    cost = cur_node['module'].get_flop_cost(input_var)
                else:
                    raise RuntimeError("Deprecated behaviour, all module should now be a NetworkBlock, got {}".format(
                        type(cur_node['module']).__name__))

                if main_cost:
                    cur_node['cost'] = cost

                self.costs[self.path_recorder.node_index[node]] = torch.Tensor(cost)
                cur_node['input'] = []

                for succ in graph.successors(node):
                    if 'input' not in graph.node[succ]:
                        graph.node[succ]['input'] = []
                    graph.node[succ]['input'].append(out)
