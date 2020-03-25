# -*- coding: UTF-8 -*-
# @Time    : 2019-10-29 12:15
# @File    : EdgeCostEvaluator.py
# @Author  : jian<jian@mltalker.com>
from nas.component.CostEvaluator import CostEvaluator
import torch
from nas.utils.globalval import *


class EdgeCostEvaluator(CostEvaluator):
    def get_cost(self, architectures):
        # self.costs is N x state
        costs = torch.gather(self.costs, dim=1, index=architectures.long())
        # costs = costs * architectures

        # costs = self.costs[:,1].unsqueeze(1).expand_as(architectures)
        # costs = architectures * costs
        return costs.sum(0)

    @property
    def total_cost(self):
        return self.costs.sum().item()
