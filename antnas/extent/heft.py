# -*- coding: UTF-8 -*-
# @Time    : 2020/11/5 12:58 下午
# @File    : heft.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from ctypes import cdll
import ctypes


# cur = cdll.LoadLibrary('./heft.so')
# pyarray = [index for index in range(10*10)]

# computing_cost = []
# communication_cost = []
# dag={1:(2,3,4,5,6),
#      2:(8,9),
#      3:(7,),
#      4:(8,9),
#      5:(9,),
#      6:(8,),
#      7:(10,),
#      8:(10,),
#      9:(10,),
#      10:()}
#
# task_num = 10
# communication_cost = [-1 for _ in range(10*10)]
# computing_cost = [0 for _ in range(10*3)]
# computing_cost[0*3+0] = 14
# computing_cost[0*3+1] = 16
# computing_cost[0*3+2] = 9
#
# computing_cost[1*3+0] = 13
# computing_cost[1*3+1] = 19
# computing_cost[1*3+2] = 18
#
# computing_cost[2*3+0] = 11
# computing_cost[2*3+1] = 13
# computing_cost[2*3+2] = 19
#
# computing_cost[3*3+0] = 13
# computing_cost[3*3+1] = 8
# computing_cost[3*3+2] = 17
#
# computing_cost[4*3+0] = 12
# computing_cost[4*3+1] = 13
# computing_cost[4*3+2] = 10
#
# computing_cost[5*3+0] = 13
# computing_cost[5*3+1] = 16
# computing_cost[5*3+2] = 9
#
# computing_cost[6*3+0] = 7
# computing_cost[6*3+1] = 15
# computing_cost[6*3+2] = 11
#
# computing_cost[7*3+0] = 5
# computing_cost[7*3+1] = 11
# computing_cost[7*3+2] = 14
#
# computing_cost[8*3+0] = 18
# computing_cost[8*3+1] = 12
# computing_cost[8*3+2] = 20
#
# computing_cost[9*3+0] = 21
# computing_cost[9*3+1] = 7
# computing_cost[9*3+2] = 16
#
#
# communication_cost[0*10+1] = 18
# communication_cost[0*10+2] = 12
# communication_cost[0*10+3] = 9
# communication_cost[0*10+4] = 11
# communication_cost[0*10+5] = 14
# communication_cost[1*10+7] = 19
# communication_cost[1*10+8] = 16
# communication_cost[2*10+6] = 23
# communication_cost[3*10+7] = 27
# communication_cost[3*10+8] = 23
# communication_cost[4*10+8] = 13
# communication_cost[5*10+7] = 15
# communication_cost[6*10+9] = 17
# communication_cost[7*10+9] = 11
# communication_cost[8*10+9] = 13
#
#
# ccomputing_cost = (ctypes.c_double * len(computing_cost))(*computing_cost)
# ccommunication_cost = (ctypes.c_double * len(communication_cost))(*communication_cost)
# a = cur.get(10,3,ccomputing_cost,ccommunication_cost)
# print(a)


def heft_shedule(task_num, device_num, computing_cost, communication_cost, heft_dll):
     ccomputing_cost = (ctypes.c_double * len(computing_cost))(*computing_cost)
     ccommunication_cost = (ctypes.c_double * len(communication_cost))(*communication_cost)
     span = heft_dll.get(task_num, device_num, ccomputing_cost, ccommunication_cost)
     return span
