# -*- coding: UTF-8 -*-
# @Time    : 2018/12/17 4:58 PM
# @File    : bayesian.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import random
import time
from functools import total_ordering
try:
    import Queue as queue
except ImportError:
    import queue

import numpy as np
import math
from sklearn import gaussian_process
import functools
import matplotlib.pyplot as plt
from antnas.networks.metric import *
import copy


class BayesianOptimizer(object):
    """

    gpr: A GaussianProcessRegressor for bayesian optimization.
    """
    def __init__(self, t_min, metric, kernel_lambda, beta):
        self.t_min = t_min
        self.metric = metric
        self.gp = gaussian_process.GaussianProcessRegressor()
        self.beta = beta
        self.elem_class = None
        self.pq = None

    def fit(self, x_queue, y_queue):
        self.gp.fit(x_queue, y_queue)
        self.elem_class = Elem
        if self.metric.higher_better():
            self.elem_class = ReverseElem

    def optimize_acq(self, sampilng_func, x_queue, y_queue, timeout=60):
        # Initialize the priority queue.
        self.pq = queue.PriorityQueue()
        for metric_value in y_queue:
            self.pq.put(self.elem_class(metric_value, None))
        
        start_time = time.time()
        t = 1.0
        t_min = self.t_min
        alpha = 0.9
        opt_acq = self._get_init_opt_acq_value()
        opt_model = None
        remaining_time = timeout

        while not self.pq.empty() and t > t_min and remaining_time > 0:
            elem = self.pq.get()
            # simulated annealing
            if self.metric.higher_better():
                temp_exp = min((elem.metric_value - opt_acq) / t, 1.0)
            else:
                temp_exp = min((opt_acq - elem.metric_value) / t, 1.0)

            ap = math.exp(temp_exp)
            if ap >= random.uniform(0, 1):
                # random sampling
                model_x = sampilng_func()
                
                # UCB acquisition function
                temp_acq_value = self.acq(np.expand_dims(model_x, 0))[0]
                self.pq.put(self.elem_class(temp_acq_value, model_x))

                if self._accept_new_acq_value(opt_acq, temp_acq_value):
                    opt_acq = temp_acq_value
                    opt_model = model_x

            t *= alpha
            remaining_time = timeout - (time.time() - start_time)

        return opt_model, opt_acq

    def predict(self, x):
        return self.gp.predict(x, return_std=True)

    def acq(self, x):
        # using Upper Confdence Bound
        mean, std = self.gp.predict(x, return_std=True)
        if self.metric.higher_better():
            return mean + self.beta * std
        return mean - self.beta * std

    def _get_init_opt_acq_value(self):
        if self.metric.higher_better():
            return -np.inf
        return np.inf

    def _accept_new_acq_value(self, opt_acq, temp_acq_value):
        if temp_acq_value > opt_acq and self.metric.higher_better():
            return True
        if temp_acq_value < opt_acq and not self.metric.higher_better():
            return True
        return False


@total_ordering
class Elem:
    def __init__(self, metric_value, graph):
        self.graph = graph
        self.metric_value = metric_value

    def __eq__(self, other):
        return self.metric_value == other.metric_value

    def __lt__(self, other):
        return self.metric_value < other.metric_value


class ReverseElem(Elem):
    def __lt__(self, other):
        return self.metric_value > other.metric_value


if __name__ == '__main__':
    class AA(object):
        def __init__(self):
            pass

        def random(self):
            return np.array([(random.random() * 2 - 1)*20])

    target_func = lambda x: x**2

    bo = BayesianOptimizer(0.000000001, BOLoss(), 0.1, 2.576)
    x = [np.array([(random.random()*2-1)*20]) for _ in range(100)]
    y = [target_func(xi[0]) for xi in x]
    bo.fit(x, y)

    plt.xlabel('Function 1', fontsize=15)
    plt.ylabel('Function 2', fontsize=15)
    plt.scatter([xi[0] for xi in x], y, c='b')

    aa = AA()

    suggestion_val_list = []
    suggestion_gt_list = []
    
    for _ in range(5):
        suggestion_val, suggestion_score_predicted = bo.optimize_acq(aa.random,x,y)
        gt_val = target_func(suggestion_val)
        suggestion_val_list.append(suggestion_val)
        suggestion_gt_list.append(gt_val)
        print('predict %f gt %f'%(suggestion_score_predicted, gt_val))

    plt.scatter([x_val[0] for x_val in x],y)
    plt.scatter(suggestion_val_list, suggestion_gt_list, c='r')

    plt.show()