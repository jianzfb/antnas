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
from tqdm import tqdm
from torch import optim
import argparse
from nas.networks.FixedNetwork import *
from nas.dataset.datasets import get_data


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def argument_parser():
    parser = argparse.ArgumentParser(description='Fixed Network Train')

    # Training
    parser.add_argument('-path', default='/Users/jian/Downloads/dataset', type=str,
                        help='path for the execution')

    parser.add_argument('-dset', default='CIFAR10', type=str, help='Dataset')
    parser.add_argument('-bs', action='store', default=2, type=int, help='Size of each batch')
    parser.add_argument('-epochs', action='store', default=300, type=int,
                        help='Number of training epochs')
    parser.add_argument('-optim', action='store', default='SGD', type=str,
                        help='Optimization method')
    parser.add_argument('-lr', action='store', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('-cuda', action='store', default='', type=str,
                        help='Enables cuda and select device')

    parser.add_argument('-draw_env', default='FIXED NETWORK', type=str, help='Visdom drawing environment')
    parser.add_argument('-wd', dest='weight_decay', action='store', default=1e-4, type=float,
                        help='weight decay used during optimisation')
    parser.add_argument('-momentum', action='store', default=0.9, type=float,
                        help='momentum used by the optimizer')
    parser.add_argument('-nesterov', action='store', default=False, type=bool,
                        help='Use Nesterov for SGD momentum')

    return parser.parse_known_args()[0]


class OutLayer(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, out_shape, in_chan=160, bias=True):
        super(OutLayer, self).__init__()
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.conv_1 = nn.Conv2d(in_chan, 960, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn = nn.BatchNorm2d(960)

        self.conv_2 = nn.Conv2d(960, 1280, kernel_size=1, stride=1, padding=0, bias=bias)
        self.conv_3 = nn.Conv2d(1280, out_shape[0], kernel_size=1, stride=1, padding=0, bias=bias)
        self.out_shape = out_shape
        self.params = {
            'module_list': ['OutLayer'],
            'name_list': ['OutLayer'],
            'OutLayer': {'out_shape': out_shape, 'in_chan': in_chan},
            'out': 'outname'
        }

    def forward(self, x, sampling=None):
        x = self.conv_1(x)
        x = self.bn(x)
        x = F.relu6(x)

        x = self.global_pool(x)
        x = self.conv_2(x)
        x = F.relu6(x)

        x = self.conv_3(x)
        return x.view(-1, *self.out_shape)

    def get_flop_cost(self, x):
        return [0] + [0] * (self.state_num - 1)


def main(args, plotter):
    if torch.cuda.is_available():
        print('CUDA is OK')

    # start run and set environment
    os.environ['CUDA_VISIBLE_DEVICES'] = args['cuda']

    # 获得数据集
    train_loader, val_loader, test_loader, data_properties = \
        get_data(args['dset'], args['bs'], args['path'], args)

    xp = mlogger.Container()
    xp.config = mlogger.Config(plotter=plotter)
    xp.epoch = mlogger.metric.Simple()

    xp.train = mlogger.Container()
    xp.train.classif_loss = mlogger.metric.Average(plotter=plotter, plot_title="classif_loss", plot_legend="train")
    xp.train.accuracy = mlogger.metric.Average(plotter=plotter, plot_title="accuracy", plot_legend="train")
    xp.train.objective_cost = mlogger.metric.Average(plotter=plotter, plot_title="objective_cost", plot_legend="architecture")
    xp.train.learning_rate = mlogger.metric.Simple(plotter=plotter, plot_title="LR", plot_legend="train")
    xp.train.timer = mlogger.metric.Timer(plotter=plotter, plot_title="Time", plot_legend="train")

    xp.val = mlogger.Container()
    xp.val.accuracy = mlogger.metric.Simple(plotter=plotter, plot_title="val_test_accuracy", plot_legend="val")
    xp.val.timer = mlogger.metric.Timer(plotter=plotter, plot_title="Time", plot_legend="val")

    xp.test = mlogger.Container()
    xp.test.accuracy = mlogger.metric.Simple(plotter=plotter, plot_title="val_test_accuracy", plot_legend="test")
    xp.test.timer = mlogger.metric.Timer(plotter=plotter, plot_title="Time", plot_legend="test")

    # 配置网络模型
    model = FixedNetwork(architecture='/Users/jian/PycharmProjects/minas/supernetwork/accuray_0.8108_flops_9814212_params_1028592.architecture',
                         output_layer_cls=OutLayer)

    # set model input
    x = torch.Tensor()
    y = torch.LongTensor()

    if len(args['cuda']) > 0 and len(args['cuda'].split(',')) > 0:
        logger.info('Running with cuda (GPU {})'.format(args['cuda']))
        model.to(int(args['cuda'].split(',')[0]))
        model = nn.DataParallel(model, device_ids=[int(i) for i in args['cuda'].split(',')])
        x = x.cuda()
        y = y.cuda()
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

    # 配置学习率
    decayRate = 0.96
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

    for epoch in range(args['epochs']):
        # write logger
        logger.info(epoch)

        # training process
        model.train()
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc='Train', ascii=True)):
            # set model status (train)
            x.resize_(inputs.size()).copy_(inputs)
            y.resize_(labels.size()).copy_(labels)

            loss, accuracy = model(x, y)
            loss = loss.mean()
            accuracy = accuracy.sum()

            # record model loss
            xp.train.classif_loss.update(loss.item())
            # record model accuracy
            xp.train.accuracy.update(accuracy * 100 / float(inputs.size(0)))

            # compute gradients
            optimizer.zero_grad()
            loss.backward()

            # update parameter
            optimizer.step()

            # record lr
            xp.train.learning_rate.update(lr_scheduler.get_lr()[0])

            # fresh log
            xp.train.timer.update()
            for metric in xp.train.metrics():
                metric.log()

            plotter.update_plots()

        # adjust learning rate
        if (epoch+1) % 30 == 0:
            lr_scheduler.step()

        # test process
        model.eval()
        # test process on val dataset
        # test process on test dataset
        if (epoch + 1) % 10 == 0:
            logger.info('\nEvaluation')
            total_correct = 0
            total = 0
            for i, (inputs, labels) in enumerate(tqdm(test_loader, desc='Test', ascii=True)):
                x.resize_(inputs.size()).copy_(inputs)
                y.resize_(labels.size()).copy_(labels)

                with torch.no_grad():
                    loss, accuracy = model(x, y)
                    total_correct += accuracy.sum()
                    total += labels.size(0)

            test_accuracy = 100 * total_correct.float().item() / total
            xp.test.accuracy.update(test_accuracy)
            xp.test.timer.update()

            for metric in xp.test.metrics():
                metric.log()

            plotter.update_plots()

        # save model state
        if (epoch + 1) % 5 == 0:
            if not os.path.exists("./supernetwork"):
                os.makedirs("./supernetwork")

            path = os.path.join("./supernetwork", "check")
            torch.save(model.state_dict(), '%s.supernet.model' % path)


if __name__ == '__main__':
    logger.info('Executing main from {}'.format(os.getcwd()))
    args = vars(argument_parser())

    plotter = mlogger.VisdomPlotter({'env': args['draw_env'],
                                     'server': 'http://localhost',
                                     'port': 8097},
                                    manual_update=True)
    main(args, plotter)
