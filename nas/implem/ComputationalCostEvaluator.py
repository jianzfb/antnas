import logging
import torch
from nas.implem.EdgeCostEvaluator import EdgeCostEvaluator
from nas.interfaces.NetworkBlock import NetworkBlock

logger = logging.getLogger(__name__)


class ComputationalCostEvaluator(EdgeCostEvaluator):
    def init_costs(self, model, main_cost):
        with torch.no_grad():
            # input_var = (torch.ones(1, *model.input_size),)
            graph = model.net

            # set costs
            self.costs = torch.Tensor(graph.number_of_nodes(), NetworkBlock.state_num)

            for node in model.traversal_order:
                cur_node = graph.node[node]
                # if isinstance(cur_node['module'], NetworkBlock):
                #     cost = cur_node['module'].get_flop_cost()
                # else:
                #     raise RuntimeError("Deprecated behaviour, all module should now be a NetworkBlock, got {}".format(
                #         type(cur_node['module']).__name__))
                cost = model.blocks[cur_node['module']].get_flop_cost()

                if main_cost:
                    cur_node['cost'] = cost

                self.costs[self.path_recorder.node_index[node]] = torch.Tensor(cost)
