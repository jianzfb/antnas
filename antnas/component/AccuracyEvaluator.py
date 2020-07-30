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
    thread_pool = None
    process_queue = queue.Queue()

    def __init__(self, *args, **kwargs):
        super(AccuracyEvaluator, self).__init__()
        self.lock = threading.Lock()

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

    @staticmethod
    def launch_thread_pool(thread_num):
        AccuracyEvaluator.thread_pool = \
            [threading.Thread(target=AccuracyEvaluator.process_func) for _ in range(thread_num)]
        for thread_index in range(thread_num):
            AccuracyEvaluator.thread_pool[thread_index].start()

    @staticmethod
    def process_func():
        while True:
            data = AccuracyEvaluator.process_queue.get()
            if data is None:
                break

            handler, handler_data = data
            handler._caculate_in_thread(*handler_data, lock=handler.lock)

    @staticmethod
    def stop():
        for _ in range(len(AccuracyEvaluator.thread_pool)):
            AccuracyEvaluator.process_queue.put(None)

        for ti in range(len(AccuracyEvaluator.thread_pool)):
            AccuracyEvaluator.thread_pool[ti].join()