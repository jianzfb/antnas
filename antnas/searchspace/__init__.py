# -*- coding: UTF-8 -*-
# @Time    : 2019-08-19 18:11
# @File    : __init__.py.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from pkgutil import walk_packages
import os


def _global_import(name):
  p = __import__(name, globals(), locals(), level=1)
  globals().pop(name)
  lst = p.__all__ if '__all__' in dir(p) else []
  print(name)
  for k in lst:
    # add global varaible
    globals()[k] = p.__dict__[k]


for _, module_name, _ in walk_packages([os.path.dirname(__file__)]):
    _global_import(module_name)