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
from antnas.networks.MNv2FixedNetwork import *
from antnas.utils.adjust import *
from antnas.utils.argument_parser import *
import math
from antnas.component.ClassificationAccuracyEvaluator import *
from OutLayerFactory import *


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageNetOutLayer(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, in_chan, **kwargs):
        super(ImageNetOutLayer, self).__init__()
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.conv_1 = nn.Conv2d(in_chan, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_1 = nn.BatchNorm2d(960,
                                   momentum=1.0 if not NetworkBlock.bn_moving_momentum else 0.1,
                                   track_running_stats=NetworkBlock.bn_track_running_stats)

        self.conv_2 = nn.Conv2d(960, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.classifier = nn.Linear(1280, 1000)
        self.dropout = torch.nn.Dropout(p=0.9)

        self.params = {
            'module_list': ['ImageNetOutLayer'],
            'name_list': ['ImageNetOutLayer'],
            'ImageNetOutLayer': {'in_chan': in_chan},
            'out': 'outname',
            'in_chan': in_chan,
        }

    def forward(self, x, sampling=None):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = F.relu(x)

        x = self.global_pool(x)
        x = self.conv_2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

@antnas_argment
def main(*args, **kwargs):
    if torch.cuda.is_available():
        logging.info("CUDA is AVAILABLE")

    # start run and set environment
    os.environ['CUDA_VISIBLE_DEVICES'] = kwargs['cuda']

    # 获得数据集
    kwargs.update({'img_size': 224, 'in_channels': 3, 'out_channels': 1000})
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

    # 配置网络损失函数
    criterion = None
    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    # 配置网络模型
    # model = FixedNetwork(architecture=kwargs['architecture'],
    #                      output_layer_cls=ImageNetOutLayer,
    #                      accuracy_evaluator_cls=lambda: ClassificationAccuracyEvaluator(),
    #                      network_name='heterogeneous-nas')
    model = MNV2FixedNetwork(architecture=kwargs['architecture'],
                         output_layer_cls=ImageNetOutLayer,
                         accuracy_evaluator_cls=lambda: ClassificationAccuracyEvaluator(topk=(1,5)),
                         network_name='heterogeneous-nas-4')
    xp.architecture.update(kwargs['architecture'])

    # 配置数据并行环境
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

    # 配置模型优化器
    optimizer = initialize_optimizer('train', parallel_model, kwargs)

    best_test_accuracy = 0.0
    lr = 0.0
    for epoch in range(kwargs['epochs']):
        # write logger
        logger.info(epoch)

        # training process
        parallel_model.train()
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc='Train', ascii=True)):
            # adjust learning rate
            adjust_lr(kwargs, optimizer, epoch, i, len(train_loader))

            # set model status (train)
            x = inputs
            y = labels

            output = parallel_model(x, y)
            loss = criterion(output, y)

            # record model loss
            xp.train.classif_loss.update(loss.item())

            # reset grad zero
            optimizer.zero_grad()
            # compute gradients
            loss.backward()
            # update parameter
            optimizer.step()

            # record lr
            lr = optimizer.param_groups[0]['lr']
            xp.train.learning_rate.update(lr)

            mlogger.update()

        # test process on test dataset
        if epoch % 1 == 0:
            logger.info('\nEvaluation')

            # test process
            parallel_model.eval()

            AccuracyEvaluator.launch_thread_pool(1)
            accuracy_evaluator = model.accuracy_evaluator()
            for i, (inputs, labels) in enumerate(tqdm(test_loader, desc='Test', ascii=True)):
                x.resize_(inputs.size()).copy_(inputs)
                y.resize_(labels.size()).copy_(labels)

                with torch.no_grad():
                    output = parallel_model(x, y)

                processed_data = accuracy_evaluator.preprocess(output, y)
                accuracy_evaluator.caculate(*processed_data)

            AccuracyEvaluator.stop()
            test_accuracy_top_1, test_accuracy_top_5 = accuracy_evaluator.accuracy()
            xp.test.accuracy.update(test_accuracy_top_1)
            mlogger.update()

            # print log
            print('epoch %d, lr %f, top1 %f, top5 %f'%(
                  epoch, (float)(lr), (float)(test_accuracy_top_1), (float)(test_accuracy_top_5)))

            # save best model state
            if best_test_accuracy < test_accuracy_top_1:
                if not os.path.exists("./supernetwork"):
                    os.makedirs("./supernetwork")

                path = os.path.join("./supernetwork", "check")
                torch.save(model.state_dict(), '%s.supernet.model' % path)
                best_test_accuracy = test_accuracy_top_1
                print('best accuracy %f'%best_test_accuracy)


if __name__ == '__main__':
    logger.info('Executing main from {}'.format(os.getcwd()))

    # 运行
    main()
