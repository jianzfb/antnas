# -*- coding: UTF-8 -*-
# @Time    : 2019-09-14 19:25
# @File    : SearchSpace.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from nas.searchspace import *


class SearchSpace(object):
    def __init__(self, arch, **kwargs):
        assert (arch in globals())
        self.model_cls = globals()[arch]

    def build(self, **kwargs):
        return self.model_cls(**kwargs)
