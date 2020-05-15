# -*- coding: UTF-8 -*-
# @Time    : 2020-03-30 23:18
# @File    : train.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import logging
import mlogger
import os
import torch
from torch import optim
from tqdm import tqdm
import argparse
from nas.dataset.datasets import get_data
from nas.networks.FixedNetwork import *
from OutLayerFactory import *
from math import cos, pi


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def argument_parser():
    parser = argparse.ArgumentParser(description='Fixed Network Train')

    # Training
    parser.add_argument('-path', default='/Users/jian/Downloads/dataset', type=str,
                        help='path for the execution')

    parser.add_argument('-dset', default='ImageNet', type=str, help='Dataset')
    parser.add_argument('-bs', action='store', default=2, type=int, help='Size of each batch')
    parser.add_argument('-epochs', action='store', default=150, type=int,
                        help='Number of training epochs')
    parser.add_argument('-optim', action='store', default='SGD', type=str,
                        help='Optimization method')
    parser.add_argument('-lr', action='store', default=0.05, type=float, help='Learning rate')
    parser.add_argument('-cuda', action='store', default='', type=str,
                        help='Enables cuda and select device')
    parser.add_argument('--lr_decay', type=str, default='cos',
                        help='mode for learning rate decay')
    parser.add_argument('-draw_env', default='PK-MobileNetV2', type=str, help='Visdom drawing environment')
    parser.add_argument('-wd', dest='weight_decay', action='store', default=0.00004, type=float,
                        help='weight decay used during optimisation')
    parser.add_argument('-momentum', action='store', default=0.9, type=float,
                        help='momentum used by the optimizer')
    parser.add_argument('-nesterov', action='store', default=False, type=bool,
                        help='Use Nesterov for SGD momentum')
    parser.add_argument('-architecture', action='store', default="./supernetwork/pk_pk_mobilenetv2.architecture", type=str, help="architecture path")
    parser.add_argument('--warmup', action='store_true',
                        help='set lower initial learning rate to warm up the training')
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--epochs_drop', type=int, default=10, help='for step mode')
    
    return parser.parse_known_args()[0]


def adjust_learning_rate(args, optimizer, epoch, iteration, num_iter):
    lr = optimizer.param_groups[0]['lr']

    warmup_epoch = 5 if args['warmup'] else 0
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = args['epochs'] * num_iter

    if args['lr_decay'] == 'step':
        lr = args['lr'] * (args['gamma'] ** ((current_iter - warmup_iter) // (args['epochs_drop']*num_iter - warmup_iter)))
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
        lr = args['lr'] * current_iter / warmup_iter

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


def main(args, plotter):
    if torch.cuda.is_available():
        logging.info("CUDA is AVAILABLE")

    # start run and set environment
    os.environ['CUDA_VISIBLE_DEVICES'] = args['cuda']

    # 获得数据集
    train_loader, val_loader, test_loader, data_properties = \
        get_data(args['dset'], args['bs'], args['path'], args)
    
    # logger
    xp = mlogger.Container()
    xp.config = mlogger.Config(plotter=plotter)
    xp.epoch = mlogger.metric.Simple()

    xp.train = mlogger.Container()
    xp.train.classif_loss = mlogger.metric.Average(plotter=plotter, plot_title="classif_loss", plot_legend="train")
    xp.train.accuracy = mlogger.metric.Average(plotter=plotter, plot_title="accuracy", plot_legend="train")
    xp.train.learning_rate = mlogger.metric.Simple(plotter=plotter, plot_title="LR", plot_legend="train")
    xp.train.timer = mlogger.metric.Timer(plotter=plotter, plot_title="Time", plot_legend="train")

    xp.test = mlogger.Container()
    xp.test.accuracy = mlogger.metric.Simple(plotter=plotter, plot_title="val_test_accuracy", plot_legend="test")
    xp.test.timer = mlogger.metric.Timer(plotter=plotter, plot_title="Time", plot_legend="test")

    NetworkBlock.bn_moving_momentum = True
    NetworkBlock.bn_track_running_stats = True
    
    # 配置网络模型
    model = FixedNetwork(architecture=args['architecture'],
                         output_layer_cls=ImageNetOutLayer,
                         plotter=plotter)

    # set model input
    x = torch.Tensor()
    y = torch.LongTensor()

    if len(args['cuda']) > 0 and len(args['cuda'].split(',')) > 0:
        logger.info('Running with cuda (GPU {})'.format(args['cuda']))
        model.to(int(args['cuda'].split(',')[0]))
        model = nn.DataParallel(model, device_ids=[int(i) for i in args['cuda'].split(',')])
        x = x.cuda(torch.device('cuda:%d'%int(args['cuda'].split(',')[0])))
        y = y.cuda(torch.device('cuda:%d'%int(args['cuda'].split(',')[0])))
    else:
        logger.warning('Running *WITHOUT* cuda')

    # 配置优化器
    optimizer = None
    if args['optim'] == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=args['lr'],
                              weight_decay=args['weight_decay'],
                              momentum=args['momentum'],
                              nesterov=args['nesterov'])
    elif args['optim'] == 'ADAM':
        optimizer = optim.Adam(model.parameters(),
                               lr=args['lr'],
                               weight_decay=args['weight_decay'])
    elif args['optim'] == 'RMS':
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=args['lr'],
                                  weight_decay=args['weight_decay'],
                                  momentum=args['momentum'])
    else:
        raise RuntimeError

    best_test_accuracy = 0.0
    for epoch in range(args['epochs']):
        # write logger
        logger.info(epoch)

        # training process
        model.train()
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc='Train', ascii=True)):
            # adjust learning rate
            adjust_learning_rate(args, optimizer, epoch, i, len(train_loader))
            
            # set model status (train)
            x.resize_(inputs.size()).copy_(inputs)
            y.resize_(labels.size()).copy_(labels)

            optimizer.zero_grad()
            loss, accuracy = model(x, y)
            loss = loss.mean()
            accuracy = accuracy.sum()

            # record model loss
            xp.train.classif_loss.update(loss.item())
            # record model accuracy
            xp.train.accuracy.update(accuracy * 100 / float(inputs.size(0)))

            # compute gradients
            loss.backward()

            # update parameter
            optimizer.step()

            # record lr
            lr = optimizer.param_groups[0]['lr']
            xp.train.learning_rate.update(lr)

            # fresh log
            xp.train.timer.update()
            for metric in xp.train.metrics():
                metric.log()

            plotter.update_plots()

        # test process on test dataset
        if (epoch + 1) % 5 == 0:
            logger.info('\nEvaluation')

            # test process
            model.eval()
            total_correct = 0
            total = 0
            for i, (inputs, labels) in enumerate(tqdm(test_loader, desc='Test', ascii=True)):
                x.resize_(inputs.size()).copy_(inputs)
                y.resize_(labels.size()).copy_(labels)

                with torch.no_grad():
                    _, accuracy = model(x, y)
                    total_correct += accuracy.sum()
                    total += labels.size(0)

            test_accuracy = 100 * total_correct.float().item() / total
            xp.test.accuracy.update(test_accuracy)
            xp.test.timer.update()

            for metric in xp.test.metrics():
                metric.log()

            plotter.update_plots()

            # save best model state
            if best_test_accuracy < test_accuracy:
                if not os.path.exists("./supernetwork"):
                    os.makedirs("./supernetwork")

                path = os.path.join("./supernetwork", "check")
                torch.save(model.state_dict(), '%s.supernet.model' % path)
                best_test_accuracy = test_accuracy


if __name__ == '__main__':
    logger.info('Executing main from {}'.format(os.getcwd()))
    args = vars(argument_parser())

    plotter = mlogger.VisdomPlotter({'env': args['draw_env'],
                                     'server': 'http://localhost',
                                     'port': 8097},
                                    manual_update=True)
    main(args, plotter)
