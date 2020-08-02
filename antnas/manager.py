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
from antnas.searchspace.SearchSpace import *
from math import cos, pi
import math
import queue
import threading
from antnas.utils.adjust import *


class Manager(object):
    def __init__(self, args, data_properties, out_layer):
        self.args = args
        self._search_space = SearchSpace(arch=args['arch'])
        assert(self._search_space is not None)

        self._model = None
        self._supernetwork = None
        self._data_properties = data_properties
        self._out_layer = out_layer

        # 模型优化器
        self._optimizer = None
        self.cuda_list = []
        self.arctecture_queue = queue.Queue(128)
        self.arctecture_sampling_thread_list = None
        self.arctecture = None

    def initialize_optimizer(self):
        if self._optimizer is not None:
            return self._optimizer

        # 初始化优化器
        self._optimizer = initialize_optimizer('search', self.parallel, self.args)
        return self._optimizer

    def initialize(self):
        # 1.step 初始化优化器
        self.initialize_optimizer()
        
        # 2.step 初始化模型参数
        initialize_weights(self._model)

        # 3.step 初始化结构采样线程
        self.arctecture_sampling_thread_list = \
            [threading.Thread(target=self.__samplingArcFunc, daemon=True) for _ in range(2)]
        
    def __samplingArcFunc(self):
        while True:
            arc = self._supernetwork.sample_arch()
            if arc is None:
                continue

            self.arctecture_queue.put(arc)
    
    def launchSamplingArcProcess(self):
        for t in self.arctecture_sampling_thread_list:
            t.start()
    
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
        search_space_args.update({'out_layer': self._out_layer})
        self._model = self._search_space.build(**search_space_args)
        self._supernetwork = self._model

        # initialize
        self.initialize()
        
        # load checkpoint
        if state_dict_path is not None:
            self._model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))

    def train(self, x, y, epoch=None, warmup=False, index=None):
        if not self.parallel.training:
            self.parallel.train()
        
        # sampling architecture
        if self.arctecture is None:
            if torch.cuda.is_available():
                self.arctecture = torch.Tensor().cuda(torch.device("cuda:%d"%self.cuda_list[0]))
            else:
                self.arctecture = torch.Tensor()

        if torch.cuda.is_available():
            sampling_arc = [self.arctecture_queue.get() for _ in range(len(self.cuda_list))]
            self.arctecture.resize_(len(self.cuda_list), len(sampling_arc[0])).copy_(torch.as_tensor(sampling_arc))
        else:
            sampling_arc = [self.arctecture_queue.get()]
            self.arctecture.resize_(1, len(sampling_arc[0])).copy_(torch.as_tensor(sampling_arc))
        
        # 1.step forward model
        loss, _, a, b = \
            self.parallel(x, y, self.arctecture, epoch=epoch, warmup=warmup)

        # 2.step get last sampling
        if loss is not None:
            loss = loss.mean()

        return loss, None, a, b

    def eval(self, x, y, loader, name=''):
        # 使用单卡计算
        if self.parallel.training:
            self.parallel.eval()

        # sampling architecture
        if self.arctecture is None:
            if torch.cuda.is_available():
                self.arctecture = torch.Tensor().cuda(torch.device("cuda:%d"%self.cuda_list[0]))
            else:
                self.arctecture = torch.Tensor()

        # random sampling architecture
        if torch.cuda.is_available():
            sampling_arc = [self.arctecture_queue.get() for _ in range(len(self.cuda_list))]
            self.arctecture.resize_(len(self.cuda_list), len(sampling_arc[0])).copy_(torch.as_tensor(sampling_arc))
        else:
            sampling_arc = [self.arctecture_queue.get()]
            self.arctecture.resize_(1, len(sampling_arc[0])).copy_(torch.as_tensor(sampling_arc))

        # 跑测试集，获得评估精度
        accuracy_evaluator = self.supernetwork.accuracy_evaluator()
        for images, labels in tqdm(loader, desc=name, ascii=True):
            x.resize_(images.size()).copy_(images)
            y.resize_(labels.size()).copy_(labels)

            with torch.no_grad():
                _, model_out, _, _ = self.parallel(x, y, self.arctecture)

            # 统计模型精度
            self.supernetwork.caculate(model_out, y, accuracy_evaluator)

        # 获得模型精度
        accuracy = self.supernetwork.accuracy(accuracy_evaluator)
        return accuracy

    @property
    def parallel(self):
        return self._model

    @parallel.setter
    def parallel(self, val):
        self._model = val

    @property
    def supernetwork(self):
        return self._supernetwork

    def adjust_lr(self, args, epoch, iteration, num_iter, except_groups=None):
        lr = adjust_lr(args, self.optimizer, epoch, iteration, num_iter, except_groups)
        return lr

    def cuda(self, cuda_list):
        self.supernetwork.to(cuda_list[0])
        if len(cuda_list) > 1:
            self.parallel = nn.DataParallel(self.supernetwork, [i for i in cuda_list])
            
        self.cuda_list = cuda_list