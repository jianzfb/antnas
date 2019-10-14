# -*- coding: UTF-8 -*-
# @Time    : 2019-08-19 18:22
# @File    : nas_main.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import argparse
import logging
import os

import mlogger
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from nas.dataset.datasets import get_data
from nas.models.NasModel import *
from nas.utils.misc import *
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def argument_parser():
    parser = argparse.ArgumentParser(description='Budgeted Super Networks')

    # Experience
    parser.add_argument('-exp-name', action='store', default='', type=str, help='Experience Name')
    # Model
    parser.add_argument('-arch', action='store', default='BiSegSN', type=str)
    parser.add_argument('-deter_eval', action='store', default=False, type=bool,
                        help='Take blocks with probas >0.5 instead of sampling during evaluation')

    # Training
    parser.add_argument('-path', default='/Users/jian/Downloads/pascal_voc/', type=str,
                        help='path for the execution')

    parser.add_argument('-dset', default='PASCAL2012SEG', type=str, help='Dataset')
    parser.add_argument('-bs', action='store', default=2, type=int, help='Size of each batch')
    parser.add_argument('-epochs', action='store', default=300, type=int,
                        help='Number of training epochs')

    parser.add_argument('-optim', action='store', default='SGD', type=str,
                        help='Optimization method')
    parser.add_argument('-nesterov', action='store', default=False, type=bool,
                        help='Use Nesterov for SGD momentum')
    parser.add_argument('-lr', action='store', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('-path_lr', action='store', default=1e-3, type=float, help='path learning rate')
    parser.add_argument('-momentum', action='store', default=0.9, type=float,
                        help='momentum used by the optimizer')
    parser.add_argument('-wd', dest='weight_decay', action='store', default=1e-4, type=float,
                        help='weight decay used during optimisation')

    parser.add_argument('-latest_num', action='store', default=1, type=int,help='save the latest number model state')
    parser.add_argument('-lr_pol_tresh', action='store', default=[150, 225], type=str,
                        help='learning rate decay rate')
    parser.add_argument('-lr_pol_val', action='store', nargs='*', default=[0.1, 0.01, 0.001], type=str,
                        help='learning rate decay period')

    parser.add_argument('-cuda', action='store', default='', type=str,
                        help='Enables cuda and select device')
    parser.add_argument('-latency', action='store', default='./latency.gpu.855.lookuptable.json',type=str,
                        help='latency lookup table')

    parser.add_argument('-draw_env', default='test', type=str, help='Visdom drawing environment')

    parser.add_argument('-regularizer', action='store', default=0, type=int, help='architecture regularizer')
    parser.add_argument('-static_proba', action='store', default=-1, type=restricted_float(0, 1),
                        help='sample a static binary weight with given proba for each stochastic Node.')

    parser.add_argument('-np', '--n_parallel', dest='n_parallel', action='store', default=3, type=int,
                        help='Maximum number of module evaluation in parallel')
    parser.add_argument('-ce', '-cost_evaluation', dest='cost_evaluation', action='store',
                        default=['comp'],
                        type=restricted_list('comp', 'latency', 'para'))
    parser.add_argument('-co', dest='cost_optimization', action='store', default='comp',
                        type=restricted_str('comp', 'latency', 'para'))

    parser.add_argument('-lambda', dest='lambda', action='store', default=1e-7, type=float,
                        help='Constant balancing the ratio classifier loss/architectural loss')
    parser.add_argument('-oc', dest='objective_cost', action='store', default=6000000, type=float,
                        help='Maximum allowed cost for architecture')
    parser.add_argument('-om', dest='objective_method', action='store', default='max',
                        type=restricted_str('max', 'abs'), help='Method used to compute the cost of an architecture')
    parser.add_argument('-pen', dest='arch_penalty', action='store', default=0, type=float,
                        help='Penalty for inconsistent architecture')
    return parser.parse_known_args()[0]


def main(args, plotter):
    # start run and set environment
    logger.info('Starting run : {}'.format(args['exp_name']))
    os.environ['CUDA_VISIBLE_DEVICES'] = args['cuda']

    # 获得数据集
    train_loader, val_loader, test_loader, data_properties = get_data(args['dset'], args['bs'], args['path'], args)
    lp = len(train_loader)

    # 创建NAS模型
    nas_model = NasModel(args, data_properties)
    # nas_model.build(n_layer=3, n_chan=32)
    # nas_model.build(blocks_per_stage=[1, 1, 1, 3, 3],
    #                 cells_per_block=[[3], [3], [3], [3, 3, 3], [3,3,3],[3,3,3]],
    #                 channels_per_block=[[16], [32], [64], [128, 128, 128],[256,256,256]],
    #                )

    # # SegSN and SegAsppSN
    # nas_model.build(blocks_per_stage=[1, 1, 1, 3],
    #                 cells_per_block=[[3], [3], [6], [6, 6, 3]],
    #                 channels_per_block=[[16], [32], [64], [128, 256, 512]])

    # # SegLargeKernelSN
    # nas_model.build(blocks_per_stage=[1, 1, 1, 2, 1],
    #                 cells_per_block=[[2], [3], [4], [4, 4], [4]],
    #                 channels_per_block=[[16], [24], [40], [80, 112], [160]])

    # BiSegSN
    nas_model.build(blocks_per_stage=[1, 1, 1, 3],
                    cells_per_block=[[1], [3], [3], [3, 3, 2]],
                    channels_per_block=[[32], [48], [64], [96, 160, 320]])

    # # Mobilenetv2BiSegSN
    # nas_model.build(blocks_per_stage=[1, 1, 1, 3],
    #                 cells_per_block=[[2], [3], [3], [3, 3, 2]],
    #                 channels_per_block=[[32], [48], [64], [96, 160, 320]])

    # # TFSN
    # nas_model.build(blocks_per_stage=[1, 1, 1, 2, 2],
    #                 cells_per_block=[[1], [2], [2], [3, 3], [2, 2]],
    #                 channels_per_block=[[16], [24], [40], [80, 112], [160, 180]])

    # nas_model.supernetwork.load_state_dict(torch.load('/Users/jian/Downloads/sn/nas_0.model', map_location='cpu'))
    # nas_model.supernetwork.load_static_architecture('/Users/jian/Downloads/sn/nas_0.architecture')

    # logger initialize
    xp = mlogger.Container()
    xp.config = mlogger.Config(plotter=plotter)
    xp.epoch = mlogger.metric.Simple()

    xp.train = mlogger.Container()
    xp.train.classif_loss = mlogger.metric.Average(plotter=plotter, plot_title="classif_loss", plot_legend="train")
    xp.train.accuracy = mlogger.metric.Average(plotter=plotter, plot_title="accuracy", plot_legend="train")
    xp.train.rewards = mlogger.metric.Average(plotter=plotter, plot_title="rewards",plot_legend="train")
    xp.train.timer = mlogger.metric.Timer(plotter=plotter, plot_title="Time", plot_legend="train")
    xp.train.objective_cost = mlogger.metric.Average(plotter=plotter, plot_title="objective_cost", plot_legend="architecture")

    xp.val = mlogger.Container()
    xp.val.accuracy = mlogger.metric.Simple(plotter=plotter, plot_title="val_test_accuracy", plot_legend="val")
    xp.val.timer = mlogger.metric.Timer(plotter=plotter, plot_title="Time", plot_legend="val")

    xp.test = mlogger.Container()
    xp.test.accuracy = mlogger.metric.Simple(plotter=plotter, plot_title="val_test_accuracy", plot_legend="test")
    xp.test.timer = mlogger.metric.Timer(plotter=plotter, plot_title="Time", plot_legend="test")

    for cost in args['cost_evaluation']:
        xp.train.__setattr__('train_sampled_%s'%cost,
                            mlogger.metric.Average(plotter=plotter,
                                                   plot_title='train_%s'%cost,
                                                   plot_legend="sampled_cost"))
        xp.train.__setattr__('train_pruned_%s'%cost,
                             mlogger.metric.Average(plotter=plotter,
                                                    plot_title='train_%s'%cost,
                                                    plot_legend="pruned_cost"))

        xp.val.__setattr__('eval_sampled_%s' % cost,
                           mlogger.metric.Average(plotter=plotter,
                                                  plot_title='eval_%s' % cost,
                                                  plot_legend="sampled_cost"))
        xp.val.__setattr__('eval_pruned_%s' % cost,
                           mlogger.metric.Average(plotter=plotter,
                                                  plot_title='eval_%s' % cost,
                                                  plot_legend="pruned_cost"))

    # set model input
    x = torch.Tensor()
    y = torch.LongTensor()

    if len(args['cuda']) > 0 and len(args['cuda'].split(',')) > 0:
        logger.info('Running with cuda (GPU {})'.format(args['cuda']))
        nas_model.cuda([int(c) for c in args['cuda'].split(',')])
        x = x.cuda()
        y = y.cuda()
    else:
        logger.warning('Running *WITHOUT* cuda')

    # training iterately
    for epoch in range(args['epochs']):
        # write logger
        logger.info(epoch)

        # lr_pol_tresh,lr_pol_val 负责网络参数学习率调整
        # path_lr 负责网络架构参数学习率调整
        nas_model.adjust_lr(epoch, args['lr_pol_tresh'], args['lr_pol_val'], logger, ['path'])
        nas_model.supernetwork.epoch = epoch

        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc='Train', ascii=True)):
            # set model status (train)
            x.resize_(inputs.size()).copy_(inputs)
            y.resize_(labels.size()).copy_(labels)

            # train and return predictions, loss, correct
            loss, model_accuracy, model_sampled_cost, model_pruned_cost = nas_model.train(x, y)

            model_sampled_cost = model_sampled_cost.mean()
            model_pruned_cost = model_pruned_cost.mean()

            # record architecture cost both sampled and pruned (training)
            xp.train.__getattribute__('train_sampled_%s' % args['cost_optimization']).update(model_sampled_cost.item())
            xp.train.__getattribute__('train_pruned_%s' % args['cost_optimization']).update(model_pruned_cost.item())

            # record model loss
            xp.train.classif_loss.update(loss.item())
            # record model accuracy
            xp.train.accuracy.update(model_accuracy * 100 / float(inputs.size(0)))

            # update model parameter
            nas_model.optimizer.zero_grad()
            loss.backward()

            # clip gradient （avoid exploding）
            torch.nn.utils.clip_grad_value_(nas_model.supernetwork.sampling_parameters.parameters(), 10)
            torch.nn.utils.clip_grad_value_(nas_model.supernetwork.blocks.parameters(), 10)

            # update parameter
            nas_model.optimizer.step()

            xp.train.timer.update()
            for metric in xp.train.metrics():
                metric.log()

            if (i + 1) % lp == 0:
                logger.info('\nEvaluation')

                progress = epoch + (i + 1) / len(train_loader)
                val_score = nas_model.eval(x, y, val_loader, 'validation')
                #test_score = nas_model.eval(x, y, test_loader, 'test')
                test_score = 0.0
                # record model accuracy on validation and test dataset
                xp.val.accuracy.update(val_score)
                xp.test.accuracy.update(test_score)

                msg = '[{:.2f}] Loss: {:.5f} - Cost: {:.3E} - Train: {:2.2f}% - Val: {:2.2f}% - Test: {:2.2f}%'
                logger.info(msg.format(progress,
                                       xp.train.classif_loss.value,
                                       xp.train.objective_cost.value,
                                       xp.train.accuracy.value,
                                       xp.val.accuracy.value,
                                       xp.test.accuracy.value))

                xp.val.timer.update()
                xp.test.timer.update()
                for metric in xp.val.metrics():
                    metric.log()
                for metric in xp.test.metrics():
                    metric.log()

            plotter.update_plots()

        # save model state
        nas_model.supernetwork.plot('./sn/')
        nas_model.supernetwork.save_architecture('./sn/',
                                                 'nas_%d'%(epoch%args['latest_num']))


if __name__ == '__main__':
    logger.info('Executing main from {}'.format(os.getcwd()))
    args = vars(argument_parser())

    plotter = mlogger.VisdomPlotter({'env': args['draw_env'],
                                     'server': 'http://localhost',
                                     'port': 8097},
                                    manual_update=True)
    main(args, plotter)
