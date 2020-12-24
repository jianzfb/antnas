# -*- coding: UTF-8 -*-
# @Time    : 2020/8/2 5:32 下午
# @File    : adjust.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from math import cos, pi
import math
import torch
from torch import optim
from torch import nn


def adjust_lr(args, optimizer, epoch, iteration, num_iter, except_groups=None):
    if except_groups is None:
        except_groups = []

    lr = optimizer.param_groups[0]['lr']

    warmup_epoch = args['warmup']
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = args['epochs'] * num_iter

    if args['lr_decay'] == 'step':
        lr = args['lr'] * (
                    args['gamma'] ** ((current_iter - warmup_iter) // (args['epochs_drop'] * num_iter - warmup_iter)))
    elif args['lr_decay'] == 'cos':
        lr = args['lr'] * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    elif args['lr_decay'] == 'linear':
        lr = args['lr'] * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
    elif args['lr_decay'] == 'schedule':
        count = sum([1 for s in args['schedule'] if s <= epoch])
        lr = args['lr'] * pow(args['gamma'], count)
    else:
        raise ValueError('Unknown lr mode {}'.format(args['lr_decay']))

    if epoch < warmup_epoch:
        lr = args['lr']

    if args['end_lr'] > 0:
        if lr < args['end_lr']:
            lr = args['end_lr']

    for param_group in optimizer.param_groups:
        if 'name' in param_group and param_group['name'] not in except_groups:
            param_group['lr'] = lr
        else:
            param_group['lr'] = lr

    return lr


def initialize_weights(model):
    for sub_m in model.modules():
        if isinstance(sub_m, nn.Conv2d):
            n = sub_m.kernel_size[0] * sub_m.kernel_size[1] * sub_m.out_channels
            sub_m.weight.data.normal_(0, math.sqrt(2. / n))
            if sub_m.bias is not None:
                sub_m.bias.data.zero_()
        elif isinstance(sub_m, nn.BatchNorm2d):
            sub_m.weight.data.fill_(1)
            sub_m.bias.data.zero_()
        elif isinstance(sub_m, nn.Linear):
            sub_m.weight.data.normal_(0, 0.01)
            sub_m.bias.data.zero_()


def initialize_optimizer(mode, model, kwargs):
    dd=[]
    for a in model.parameters():
        dd.append(a)
    print(len(dd))

    optimizer = None
    if kwargs['optim'] == 'SGD':
        if mode == 'search':
            # 区分路径参数和模型参数
            optimizer = optim.SGD([
                {'params': model.parameters(), 'name': 'blocks'},
                {'params': filter(lambda x: x.requires_grad, model.sampling_parameters.parameters()),
                 'name': 'path',
                 'lr': kwargs['path_lr'],
                 'momentum': False,
                 'weight_decay': 0.01}
            ], lr=kwargs['lr'],
                weight_decay=kwargs['weight_decay'],
                momentum=kwargs['momentum'],
                nesterov=kwargs['nesterov'])
        else:
            optimizer = optim.SGD(model.parameters(),
                                  lr=kwargs['lr'],
                                  weight_decay=kwargs['weight_decay'],
                                  momentum=kwargs['momentum'],
                                  nesterov=kwargs['nesterov'])
    elif kwargs['optim'] == 'ADAM':
        optimizer = optim.Adam(model.parameters(),
                               lr=kwargs['lr'],
                               weight_decay=kwargs['weight_decay'])
    elif kwargs['optim'] == 'RMS':
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=kwargs['lr'],
                                  weight_decay=kwargs['weight_decay'],
                                  momentum=kwargs['momentum'])
    else:
        raise RuntimeError

    return optimizer
