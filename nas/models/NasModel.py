# -*- coding: UTF-8 -*-
# @Time    : 2019-08-19 18:37
# @File    : NasModel.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import torch
from torch.autograd import Variable
from torch import optim

from nas.implem.ParameterCostEvaluator import ParameterCostEvaluator
from nas.implem.TimeCostEvaluator import TimeCostEvaluator
from nas.implem.BaselineSSN import BaselineSSN
from nas.implem.ComputationalCostEvaluator import ComputationalCostEvaluator
from nas.interfaces.PathRecorder import PathRecorder
from tqdm import tqdm


class NasModel(object):
    def __init__(self, args, data_properties):
        self.args = args
        # 创建搜索空间
        self._model = BaselineSSN(blocks_per_stage=[1, 1, 1, 3],
                            cells_per_block=[[3], [3], [6], [6, 6, 3]],
                            channels_per_block=[[16], [32], [64], [128, 256, 512]],
                            data_prop=data_properties,
                            static_node_proba=args['static'],
                            deter_eval=args['deter_eval'])

        # 架构路径分析
        self.path_recorder = PathRecorder(self.model.graph, self.model.out_node)
        self._model.subscribe(self.path_recorder.new_event)

        # 架构代价估计
        self._cost_evaluators = None
        # 模型优化器
        self._optimizer = None

        # 初始化架构约束和模型优化器
        self.initialize()

    def initialize_cost_evaluator(self):
        if self._cost_evaluators is not None:
            return self._cost_evaluators

        cost_evaluators = {
            'comp': ComputationalCostEvaluator,
            'time': TimeCostEvaluator,
            'param': ParameterCostEvaluator
        }

        self.model.eval()

        used_ce = {}
        for k in self.args['cost_evaluation']:
            used_ce[k] = cost_evaluators[k](path_recorder=self.path_recorder)
            used_ce[k].init_costs(self.model, main_cost=(k == self.args['cost_optimization']))

        self._cost_evaluators = used_ce
        return self._cost_evaluators

    def initialize_optimizer(self):
        if self._optimizer is not None:
            return self._optimizer

        if self.args['optim'] == 'SGD':
            optimizer = optim.SGD([
                {'params': self.model.blocks.parameters(), 'name': 'blocks'},
                {'params': filter(lambda x: x.requires_grad, self.model.sampling_parameters.parameters()),
                 'name': 'path',
                 'lr': self.args['path_lr'],
                 'momentum': False,
                 'weight_decay': 0}
            ], lr=self.args['lr'], weight_decay=self.args['weight_decay'], momentum=self.args['momentum'], nesterov=self.args['nesterov'])
        elif self.args['optim'] == 'ADAM':
            optimizer = optim.Adam(self.model.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        elif self.args['optim'] == 'RMS':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'],
                                      momentum=self.args['momentum'])
        else:
            raise RuntimeError

        self._optimizer = optimizer
        return self._optimizer

    def initialize(self):
        self.initialize_optimizer()
        self.initialize_cost_evaluator()

    @property
    def optimizer(self):
        return self._optimizer

    def train(self, x, y):
        if not self.model.training:
            self.model.train()

        # 1.step get model output feature map
        predictions = self.model(Variable(x))

        # 2.step compute loss
        loss = self.model.loss(predictions, Variable(y))

        # 3.step compute accuracy
        correct = self.model.accuray(predictions, Variable(y))

        return predictions, loss, correct

    def eval(self, x, y, loader, name=''):
        if self.model.training:
           self.model.eval()

        correct = 0
        total = 0
        for images, labels in tqdm(loader, desc=name, ascii=True):
            x.resize_(images.size()).copy_(images)
            y.resize_(labels.size()).copy_(labels)

            # 1.step get model output feature map
            with torch.no_grad():
                preds = self.model(x)

            # 2.step compute accuracy
            batch_correct = self.model.accuray(preds, y)

            correct += batch_correct
            total += labels.size(0)

        return 100 * correct.float().item() / total

    @property
    def architectures(self):
        return self.path_recorder.get_architectures(self.model.out_node)

    @property
    def architecture_consistence(self):
        return self.path_recorder.get_consistence(self.model.out_node).float()

    @property
    def architecture_cost_evaluators(self):
        return self._cost_evaluators

    @property
    def model(self):
        return self._model

    def architecture_loss(self, *args, **kwargs):
        return self.model.architecture_loss(*args, **kwargs)

    def adjust_lr(self, epoch, tresh, val, logger=None, except_groups=None):
        if except_groups is None:
            except_groups = []
        assert len(tresh) == len(val) - 1
        i = 0
        while i < len(tresh) and epoch > tresh[i]:
            i += 1
        lr = val[i]

        if logger is not None:
            logger.info('Setting learning rate to {:.5f}'.format(lr))

        for param_group in self.optimizer.param_groups:
            if param_group['name'] not in except_groups:
                param_group['lr'] = lr
            elif logger is not None:
                logger.info('{} - {}'.format(param_group['name'], param_group['lr']))

        return lr