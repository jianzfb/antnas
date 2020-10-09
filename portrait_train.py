# -*- coding: UTF-8 -*-
# @Time    : 2020-03-30 23:18
# @File    : cifar10_train.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import logging
import antvis.client.mlogger as mlogger
import os
import torch
from torch import optim
from tqdm import tqdm
import argparse
from antnas.dataset.datasets import get_data
from antnas.networks.FixedNetwork import *
from antnas.utils.argument_parser import *
from antnas.component.Loss import *
from antnas.component.SegmentationAccuracyEvaluator import *
from antnas.utils.adjust import *
from math import cos, pi
import math


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OutLayer(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, in_chan=32, out_chan=2, bias=True):
        super(OutLayer, self).__init__()
        self.conv_1 = nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=bias)
        self.conv_2 = nn.Conv2d(out_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=bias)

        self.params = {
            'module_list': ['OutLayer'],
            'name_list': ['OutLayer'],
            'OutLayer': {'out_chan': out_chan, 'in_chan': in_chan},
            'out': 'outname',
            'in_chan': in_chan,
            'out_chan': out_chan
        }

    def forward(self, x, sampling=None):
        # B x C x H x W
        x = self.conv_1(x)
        x = torch.nn.functional.interpolate(x,
                                            size=(384, 384),
                                            mode='bilinear',
                                            align_corners=True)

        x = self.conv_2(x)

        return x

    def get_flop_cost(self, x):
        return [0] + [0] * (self.state_num - 1)


@antnas_argment
def main(*args, **kwargs):
    if torch.cuda.is_available():
        logging.info("CUDA is AVAILABLE")

    # start run and set environment
    os.environ['CUDA_VISIBLE_DEVICES'] = kwargs['cuda']

    # 获得数据集
    train_loader, val_loader, test_loader, data_properties = \
        get_data(kwargs['dset'], kwargs['bs'], kwargs['path'], kwargs)
    
    # logger
    xp = mlogger.Container()
    xp.epoch = mlogger.metric.Simple(plot_title='epoch')
    xp.architecture = mlogger.complex.Text(plot_title='architecture')

    xp.train = mlogger.Container()
    xp.train.classif_loss = mlogger.metric.Average(plot_title="classif_loss")
    xp.train.learning_rate = mlogger.metric.Simple(plot_title="LR")

    xp.test = mlogger.Container()
    xp.test.accuracy = mlogger.metric.Simple(plot_title="test_accuracy")

    NetworkBlock.bn_moving_momentum = True
    NetworkBlock.bn_track_running_stats = True

    # 配置网络模型
    model = FixedNetwork(architecture=kwargs['architecture'],
                         output_layer_cls=OutLayer,
                         loss_func=cross_entropy,
                         accuracy_evaluator_cls=lambda :SegmentationAccuracyEvaluator(2, True))
    xp.architecture.update(kwargs['architecture'])

    # 初始化模型权重
    initialize_weights(model)

    # 配置优化器
    optimizer = initialize_optimizer('train', model, kwargs)

    # set model input
    x = torch.Tensor()
    y = torch.LongTensor()
    parallel_model = model
    if len(kwargs['cuda']) > 0 and len(kwargs['cuda'].split(',')) > 0:
        logger.info('Running with cuda (GPU {})'.format(kwargs['cuda']))
        model.to(int(kwargs['cuda'].split(',')[0]))
        parallel_model = nn.DataParallel(model, device_ids=[int(i) for i in kwargs['cuda'].split(',')])
        x = x.cuda(torch.device('cuda:%d'%int(kwargs['cuda'].split(',')[0])))
        y = y.cuda(torch.device('cuda:%d'%int(kwargs['cuda'].split(',')[0])))
    else:
        logger.warning('Running *WITHOUT* cuda')

    best_test_accuracy = 0.0
    for epoch in range(kwargs['epochs']):
        # write logger
        logger.info(epoch)

        # training process
        model.train()
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc='Train', ascii=True)):
            # adjust learning rate
            adjust_lr(kwargs, optimizer, epoch, i, len(train_loader))

            # set model status (train)
            x.resize_(inputs.size()).copy_(inputs)
            y.resize_(labels.size()).copy_(labels)

            optimizer.zero_grad()
            loss, _ = parallel_model(x, y)
            loss = loss.mean()

            # record model loss
            xp.train.classif_loss.update(loss.item())

            # compute gradients
            loss.backward()

            # update parameter
            optimizer.step()

            # record lr
            lr = optimizer.param_groups[0]['lr']
            xp.train.learning_rate.update(lr)

            mlogger.update()

        # test process on test dataset
        if (epoch + 1) % 5 == 0:
            logger.info('\nEvaluation')

            # test process
            parallel_model.eval()

            AccuracyEvaluator.launch_thread_pool(1)
            accuracy_evaluator = model.accuracy_evaluator()
            for i, (inputs, labels) in enumerate(tqdm(test_loader, desc='Test', ascii=True)):
                x.resize_(inputs.size()).copy_(inputs)
                y.resize_(labels.size()).copy_(labels)

                with torch.no_grad():
                    _, feature_out = parallel_model(x, y)

                    processed_data = accuracy_evaluator.preprocess(feature_out, y)
                    accuracy_evaluator.caculate(*processed_data)

            AccuracyEvaluator.stop()
            test_accuracy = accuracy_evaluator.accuracy()
            xp.test.accuracy.update(test_accuracy)
            mlogger.update()

            # save best model state
            if best_test_accuracy < test_accuracy:
                if not os.path.exists("./supernetwork"):
                    os.makedirs("./supernetwork")

                path = os.path.join("./supernetwork", "check")
                torch.save(model.state_dict(), '%s.supernet.model' % path)
                best_test_accuracy = test_accuracy


if __name__ == '__main__':
    logger.info('Executing main from {}'.format(os.getcwd()))

    # 运行
    main()
