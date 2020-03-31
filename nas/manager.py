# -*- coding: UTF-8 -*-
# @Time    : 2019-08-19 18:37
# @File    : manager.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from torch.autograd import Variable
from torch import optim
from tqdm import tqdm
from nas.searchspace.SearchSpace import *


class Manager(object):
    def __init__(self, args, data_properties):
        self.args = args
        self._search_space = SearchSpace(arch=args['arch'])
        assert(self._search_space is not None)

        self._model = None
        self._supernetwork = None
        self._data_properties = data_properties

        # 模型优化器
        self._optimizer = None

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
                 'weight_decay': 0.01}
            ], lr=self.args['lr'], weight_decay=self.args['weight_decay'], momentum=self.args['momentum'], nesterov=self.args['nesterov'])
        elif self.args['optim'] == 'ADAM':
            optimizer = optim.Adam(self.model.parameters(),
                                   lr=self.args['lr'],
                                   weight_decay=self.args['weight_decay'])
        elif self.args['optim'] == 'RMS':
            optimizer = optim.RMSprop(self.model.parameters(),
                                      lr=self.args['lr'],
                                      weight_decay=self.args['weight_decay'],
                                      momentum=self.args['momentum'])
        else:
            raise RuntimeError

        self._optimizer = optimizer
        return self._optimizer

    def initialize(self):
        self.initialize_optimizer()

    @property
    def optimizer(self):
        return self._optimizer

    def build(self, state_dict_path=None, **kwargs):
        if self._model is not None:
            return

        # build model
        search_space_args = self.args
        search_space_args.update(kwargs)
        search_space_args.update({'data_prop': self._data_properties})
        self._model = self._search_space.build(**search_space_args)
        self._supernetwork = self._model

        # initialize model
        self.initialize()

        # load checkpoint
        if state_dict_path is not None:
            self._model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))

    def train(self, x, y):
        if not self.model.training:
            self.model.train()

        # 1.step forward model
        loss, accuracy, sample_cost, prune_cost = self.model(x, y)

        # 2.step get last sampling
        return loss.mean(), \
               accuracy.sum(), \
               sample_cost.mean() if sample_cost is not None else None, \
               prune_cost.mean() if prune_cost is not None else None

    def eval(self, x, y, loader, name=''):
        if self.model.training:
           self.model.eval()

        total_correct = 0
        total = 0
        for images, labels in tqdm(loader, desc=name, ascii=True):
            x.resize_(images.size()).copy_(images)
            y.resize_(labels.size()).copy_(labels)

            with torch.no_grad():
                _, accuracy, _, _ = self.model(Variable(x), Variable(y))

            total_correct += accuracy.sum()
            total += labels.size(0)

        return 100 * total_correct.float().item() / total

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, val):
        self._model = val

    @property
    def supernetwork(self):
        return self._supernetwork

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

    def cuda(self, cuda_list):
        self.model.to(0)
        if len(cuda_list) > 1:
            self.model = nn.DataParallel(self.model, [i for i in range(len(cuda_list))])