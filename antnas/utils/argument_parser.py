# -*- coding: UTF-8 -*-
# @Time    : 2020/7/23 11:22 上午
# @File    : argument_parser.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import argparse
import antvis.client.mlogger as mlogger
from antnas.utils.misc import *


def antnas_argment(func):
    parser = argparse.ArgumentParser(description='ANTNAS')

    # Experience
    parser.add_argument('-exp-project', action='store', default='nas', type=str, help='Experiment project')
    parser.add_argument('-exp-name', action='store', default='experiment', type=str, help='Experiment name')
    parser.add_argument('-dashboard_ip', action='store', default='127.0.0.1', type=str)
    parser.add_argument('-dashboard_port', action='store', default=8999, type=int)

    # Model
    parser.add_argument('-arch', action='store', default='PKAsynImageNetSN', type=str)
    parser.add_argument('-deter_eval', action='store', default=False, type=bool,
                        help='Take blocks with probas >0.5 instead of sampling during evaluation')

    # Training
    parser.add_argument('-path', default='/Users/zhangjian52/Downloads/workspace/factory/dataset/refine', type=str,
                        help='path for the execution')

    parser.add_argument('-dset', default='PLACEHOLDER', type=str, help='Dataset')
    parser.add_argument('-bs', action='store', default=4, type=int, help='Size of each batch')
    parser.add_argument('-epochs', action='store', default=0, type=int,
                        help='Number of training epochs')
    parser.add_argument('-evo_epochs', action='store', default=1, type=int,
                        help='Number of architecture searching epochs')
    parser.add_argument('-warmup', action='store', default=0, type=int,help='warmup epochs before searching architecture')
    parser.add_argument('-iterator_search', action='store', default=False, type=bool,
                        help='is iterator search')
    parser.add_argument('-population_size', action='store', default=2, type=int,
                        help='population size for NSGAII')

    parser.add_argument('-optim', action='store', default='SGD', type=str,
                        help='Optimization method')
    parser.add_argument('-nesterov', action='store', default=False, type=bool,
                        help='Use Nesterov for SGD momentum')
    parser.add_argument('-lr', action='store', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('-end_lr', action='store', default=0.000001, type=float, help='End Learning rate')
    parser.add_argument('-path_lr', action='store', default=1e-3, type=float, help='path learning rate')
    parser.add_argument('-lr_decay', type=str, default='step', help='mode for learning rate decay')
    parser.add_argument('-schedule', type=int, nargs='+', default=[150, 225],
                        help='decrease learning rate at these epochs.')

    parser.add_argument('-gamma', type=float, default=0.9, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('-epochs_drop', type=int, default=16, help='for step mode')

    parser.add_argument('-momentum', action='store', default=0.9, type=float,
                        help='momentum used by the optimizer')
    parser.add_argument('-wd', dest='weight_decay', action='store', default=1e-4, type=float,
                        help='weight decay used during optimisation')

    parser.add_argument('-latest_num', action='store', default=1, type=int, help='save the latest number model state')
    # parser.add_argument('-lr_pol_tresh', action='store', default=[150, 225], type=str,
    #                     help='learning rate decay rate')
    # parser.add_argument('-lr_pol_val', action='store', nargs='*', default=[0.1, 0.01, 0.001], type=str,
    #                     help='learning rate decay period')
    #
    parser.add_argument('-cuda', action='store', default='', type=str,
                        help='Enables cuda and select device')
    parser.add_argument('-latency', action='store',
                        default='/Users/zhangjian52/Downloads/latency.cpu.gpu.855.224.lookuptable.json', type=str,
                        help='latency lookup table')

    parser.add_argument('-static_proba', action='store', default=-1, type=restricted_float(0, 1),
                        help='sample a static binary weight with given proba for each stochastic Node.')
    parser.add_argument('-static_arc', dest='', action='store', default=[], type=list, help='fixed architecture')

    parser.add_argument('-np', '--n_parallel', dest='n_parallel', action='store', default=3, type=int,
                        help='Maximum number of module evaluation in parallel')
    parser.add_argument('-ce', '-cost_evaluation', dest='cost_evaluation', action='store',
                        default=['latency'],
                        type=restricted_list('comp', 'latency', 'param'))
    parser.add_argument('-co', dest='cost_optimization', action='store', default='latency',
                        type=restricted_str('comp', 'latency', 'param'))
    parser.add_argument('-devices', dest='devices', help='support computing device select', type=list, default=[0,1])

    parser.add_argument('-lambda', dest='lambda', action='store', default=1e-7, type=float,
                        help='Constant balancing the ratio classifier loss/architectural loss')
    parser.add_argument('-oc', dest='objective_cost', action='store', default=60000000000, type=float,
                        help='Maximum allowed cost for architecture')
    parser.add_argument('-max_comp', dest='max_comp', action='store', default=-1, type=float,
                        help='Maximum allowed cost for architecture')
    parser.add_argument('-min_comp', dest='min_comp', action='store', default=-1, type=float,
                        help='Maximum allowed cost for architecture')

    parser.add_argument('-max_latency', dest='max_latency', action='store', default=-1, type=float,
                        help='Maximum allowed cost for architecture')
    parser.add_argument('-min_latency', dest='min_latency', action='store', default=-1, type=float,
                        help='Maximum allowed cost for architecture')

    parser.add_argument('-max_param', dest='max_param', action='store', default=-1, type=float,
                        help='Maximum allowed cost for architecture')
    parser.add_argument('-min_param', dest='min_param', action='store', default=-1, type=float,
                        help='Maximum allowed cost for architecture')

    parser.add_argument('-om', dest='objective_method', action='store', default='max',
                        type=restricted_str('max', 'abs'), help='Method used to compute the cost of an architecture')
    parser.add_argument('-pen', dest='arch_penalty', action='store', default=0, type=float,
                        help='Penalty for inconsistent architecture')
    parser.add_argument('-model_path', dest="model_path", action='store', default="", type=str)

    parser.add_argument('-anchor_archs', dest="anchor_archs", action='store', default=[], type=list)
    parser.add_argument('-anchor_states', dest="anchor_states", action='store', default=[], type=list)

    parser.add_argument('-architecture', action='store', default="/Users/zhangjian52/Downloads/check-nas/accuray_0.4946_latency_22.92_params_5513760.architecture", type=str, help="architecture path")

    kargs = vars(parser.parse_known_args()[0])

    def wrapper():
        # 配置日志系统
        mlogger.config(kargs['dashboard_ip'], kargs['dashboard_port'], kargs['exp_project'], kargs['exp_name'])
        # 运行
        func(**kargs)
        # 退出日志系统
        mlogger.exit()
        return None

    return wrapper