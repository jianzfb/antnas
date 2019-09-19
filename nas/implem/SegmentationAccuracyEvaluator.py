# -*- coding: UTF-8 -*-
# @Time    : 2019-09-14 17:22
# @File    : SegmentationAccuracyEvaluator.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from nas.interfaces.AccuracyEvaluator import *
from nas.utils.function import *
import torch


class SegmentationAccuracyEvaluator(AccuracyEvaluator):
    def __init__(self, *args, **kwargs):
        super(SegmentationAccuracyEvaluator, self).__init__(*args, **kwargs)
        self.class_num = kwargs['class_num']

    def task_type(self):
        return 'SEGMENTATION'

    def accuracy(self, preditions, labels):
        # MeanIOU
        labels = torch.where(labels==255, torch.full_like(labels, 0), labels)
        preditions = torch.nn.Softmax2d()(preditions)
        preditions_argmax = preditions.argmax(1, keepdim=True)
        preditions_argmax = make_one_hot(preditions_argmax, self.class_num)
        labels_one_hot = make_one_hot(labels.view(labels.size(0), 1, labels.size(1), labels.size(2)), self.class_num)

        eps = 1e-7
        bs = preditions.shape[0]

        bs_score = 0.0
        for index in range(bs):
            pr = preditions_argmax[index].float()
            gt = labels_one_hot[index].float()

            scores = 0.0
            scores_count = 0
            for c in range(self.class_num):
                if c == 0:
                    continue

                pr_c = pr[c]
                gt_c = gt[c]

                if gt_c.sum().long() == 0:
                    continue

                intersection = torch.sum(pr_c * gt_c)
                union = torch.sum(gt_c) + torch.sum(pr_c) - intersection + eps
                scores += (intersection + eps) / union
                scores_count += 1

            if scores_count > 0:
                scores = scores / scores_count

            bs_score = bs_score + scores

        return bs_score
