# -*- coding: UTF-8 -*-
# @Time    : 2020/11/3 8:58 上午
# @File    : threadpool.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import threading
import queue


class ThreadPool(object):
    def __init__(self, thread_num, reserve_size, **kwargs):
        self.process_queue = queue.Queue()
        self.thread_num = thread_num
        self.thread_pool = \
            [threading.Thread(target=self.process_func) for _ in range(thread_num)]
        for thread_index in range(thread_num):
            self.thread_pool[thread_index].start()
        self.result = [None for _ in range(reserve_size)]

    def add(self, *args):
        self.process_queue.put(args)

    def process_func(self):
        while True:
            data = self.process_queue.get()
            if data is None:
                break

            handler, params, index = data
            self.result[index] = handler(params)

    def stop(self):
        for _ in range(len(self.thread_pool)):
            self.process_queue.put(None)

        for ti in range(len(self.thread_pool)):
            self.thread_pool[ti].join()