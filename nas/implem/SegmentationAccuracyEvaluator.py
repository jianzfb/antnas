# -*- coding: UTF-8 -*-
# @Time    : 2019-09-14 17:22
# @File    : SegmentationAccuracyEvaluator.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from nas.interfaces.AccuracyEvaluator import *
import torch
import numpy as np


class SegmentationAccuracyEvaluator(AccuracyEvaluator):
    def __init__(self, *args, **kwargs):
        super(SegmentationAccuracyEvaluator, self).__init__(*args, **kwargs)
        self.threshold = kwargs['threshold']
        self.class_num = kwargs['class_num']

    def task_type(self):
        return 'SEGMENTATION'

    def accuracy(self, preditions, labels):
        # MeanIOU
        preditions = torch.nn.Softmax2d()(preditions)
        labels = torch.nn.functional.one_hot(torch.squeeze(labels), self.class_num)
        labels = labels.permute(0, 3, 1, 2)

        eps = 1e-7
        bs = preditions.shape[0]

        bs_score = []
        for index in range(bs):
            pr = preditions[index].float()
            gt = labels[index].float()

            scores = []
            for c in range(self.class_num):
                pr_c = (pr[c] > self.threshold).float()
                gt_c = gt[c]

                if pr_c.sum().long() == 0 and gt_c.sum().long() == 0:
                    continue

                intersection = torch.sum(pr_c * gt_c)
                union = torch.sum(gt_c) + torch.sum(pr_c) - intersection + eps
                scores.append((intersection + eps) / union)

            bs_score.append(torch.Tensor(scores).mean())

        bs_score_mean = torch.Tensor(bs_score).mean()
        return bs_score_mean
