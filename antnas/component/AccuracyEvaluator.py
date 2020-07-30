# -*- coding: UTF-8 -*-
# @Time    : 2019-08-19 19:25
# @File    : AccuracyEvaluator.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import abc
import threading
import queue

class AccuracyEvaluator(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, *args, **kwargs):
        super(AccuracyEvaluator, self).__init__()
        self.wating_queue = queue.Queue()
        thread_num = kwargs.get('thread_num', 4)
        lock = threading.Lock()
        self.process_thread_pool = [threading.Thread(target=self.process_func, args=(lock,)) for _ in range(thread_num)]
        for thread_index in range(thread_num):
            self.process_thread_pool[thread_index].start()

    @abc.abstractmethod
    def task_type(self):
        raise NotImplementedError

    @abc.abstractmethod
    def caculate(self, pred, label, ignore=None):
        raise NotImplementedError

    @abc.abstractmethod
    def accuracy(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self):
        pass

    def _caculate_in_thread(self, *args, **kwargs):
        pass

    def process_func(self, lock):
        while True:
            data = self.wating_queue.get()
            if data is None:
                break

            self._caculate_in_thread(*data, lock=lock)

    def stop(self):
        for _ in range(len(self.process_thread_pool)):
            self.wating_queue.put(None)