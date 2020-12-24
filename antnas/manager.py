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
from antnas.component.AccuracyEvaluator import *
import time


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
        self.arctecture_queue = queue.Queue()
        self.arctecture_sampling_thread_list = None
        self.state = threading.Condition()
        self.paused = False

        self.arctecture = None
        self._criterion = None
        self._accuracy_evaluator = None

    def initialize_optimizer(self):
        if self._optimizer is not None:
            return self._optimizer

        # 初始化优化器
        self._optimizer = initialize_optimizer('search', self.parallel, self.args)
        return self._optimizer

    def __initialize(self):
        # 1.step 初始化优化器
        print('[manager] initialize optimizer')
        self.initialize_optimizer()
        
        # 2.step 初始化模型参数
        print('[manager] initialize weights')
        initialize_weights(self._model)

        # 3.step 初始化结构采样线程
        print('[manager] launch architecture sampling thread')
        self.arctecture_sampling_thread_list = \
            [threading.Thread(target=self.__asyn_sampling_arc_func, daemon=True) for _ in range(1)]

        # launch arc sampling thread
        for t in self.arctecture_sampling_thread_list:
            t.start()

    def __asyn_sampling_arc_func(self):
        while True:
            with self.state:
                # 是否暂停
                if self.paused:
                    self.state.wait()

                # 判断是否超过容量
                if self.arctecture_queue.qsize() >= 128:
                    time.sleep(1)
                    continue

                # 获取采样结构
                arc = self._supernetwork.sample_arch()
                if arc is None:
                    continue

                # 将采样结构加入队列
                self.arctecture_queue.put(arc)

    def reset(self):
        # 采样线程暂停
        print('[manager/reset] pause sampling thread')
        with self.state:
            self.paused = True

        # 清空结构候选池
        print('[manager/reset] clear architecture queue')
        while not self.arctecture_queue.empty():
            self.arctecture_queue.get()

        # 采样线程启动
        print('[manager/reset] resume sampling thread')
        with self.state:
            self.paused = False
            self.state.notify()

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
        
        # load checkpoint
        if state_dict_path is not None:
            self._model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))

        self._criterion = self._model.criterion
        self._accuracy_evaluator = self._model.accuracy_evaluator

    def init(self, *args, **kwargs):
        # init supernetwork
        self._supernetwork.init(**kwargs)
        # draw
        self._supernetwork.draw()
        # init manager
        self.__initialize()

    def train(self, x, y, epoch=None, index=None):
        # 设置标记
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
        _, model_out, a, b = \
            self.parallel(x, y, self.arctecture, epoch=epoch)

        return None, model_out, a, b

    def eval(self, x, y, loader, name=''):
        # 设置标记
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
        AccuracyEvaluator.launch_thread_pool(1)
        accuracy_evaluator = self.accuracy_evaluator()
        for images, labels in tqdm(loader, desc=name, ascii=True):
            x.resize_(images.size()).copy_(images)
            y.resize_(labels.size()).copy_(labels)

            with torch.no_grad():
                _, model_out, _, _ = self.parallel(x, y, self.arctecture)

            # 统计模型精度
            processed_data = accuracy_evaluator.preprocess(model_out, y)
            accuracy_evaluator.caculate(*processed_data)

        # 获得模型精度
        AccuracyEvaluator.stop()
        accuracy = accuracy_evaluator.accuracy()
        if type(accuracy) == list or type(accuracy) == tuple:
            accuracy = accuracy[0]

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

    @property
    def criterion(self):
        return self._criterion

    @property
    def accuracy_evaluator(self):
        return self._accuracy_evaluator

    def adjust_lr(self, args, epoch, iteration, num_iter, except_groups=None):
        lr = adjust_lr(args, self.optimizer, epoch, iteration, num_iter, except_groups)
        return lr

    def cuda(self, cuda_list):
        self.supernetwork.to(cuda_list[0])
        if len(cuda_list) > 1:
            self.parallel = nn.DataParallel(self.supernetwork, [i for i in cuda_list])
            
        self.cuda_list = cuda_list