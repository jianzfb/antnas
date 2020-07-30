# -*- coding: UTF-8 -*-
# @Time    : 2019-09-14 17:22
# @File    : SegmentationAccuracyEvaluator.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import os
import sys
import numpy as np
from antnas.component.AccuracyEvaluator import *
from antnas.utils.function import *
from scipy.sparse import csr_matrix


# class SegmentationAccuracyEvaluator(AccuracyEvaluator):
#     def __init__(self, *args, **kwargs):
#         super(SegmentationAccuracyEvaluator, self).__init__(*args, **kwargs)
#         self.class_num = (int)(kwargs['class_num'])
#
#     def task_type(self):
#         return 'SEGMENTATION'
#
#     def accuracy(self, preditions, labels):
#         if (labels.shape[1] != preditions.shape[2]) or (labels.shape[2] != preditions.shape[3]):
#             preditions = F.upsample(preditions, size=(labels.shape[1], labels.shape[2]), mode='bilinear')
#
#         # MeanIOU
#         labels = torch.where(labels == 255, torch.full_like(labels, 0), labels)
#         preditions = torch.nn.Softmax2d()(preditions)
#         preditions_argmax = preditions.argmax(1, keepdim=True)
#
#         preditions_argmax = make_one_hot(preditions_argmax, self.class_num)
#         labels_one_hot = make_one_hot(labels.view(labels.size(0), 1, labels.size(1), labels.size(2)), self.class_num)
#
#         eps = 1e-7
#         bs = preditions.shape[0]
#
#         bs_score = []
#         for index in range(bs):
#             pr = preditions_argmax[index].float()
#             gt = labels_one_hot[index].float()
#
#             scores = 0.0
#             class_num = 0
#             for c in range(self.class_num):
#                 pr_c = pr[c]
#                 gt_c = gt[c]
#
#                 if gt_c.sum().long() == 0:
#                     continue
#
#                 intersection = torch.sum(pr_c * gt_c)
#                 union = torch.sum(gt_c) + torch.sum(pr_c) - intersection + eps
#                 scores += intersection / union
#                 class_num += 1
#
#             scores = scores / (class_num + eps)
#             bs_score.append(scores)
#
#         bs_score = torch.as_tensor(bs_score, device=preditions.device)
#         return bs_score


class SegmentationAccuracyEvaluator(AccuracyEvaluator):
    """
        Confusion Matrix for segmentation evaluation
    """

    def __init__(self, num_classes=2, streaming=False):
        super(SegmentationAccuracyEvaluator, self).__init__()
        self.confusion_matrix = np.zeros([num_classes, num_classes],
                                         dtype='int64')
        self.num_classes = num_classes
        self.streaming = streaming

    def task_type(self):
        return 'SEGMENTATION'

    def reset(self):
        self.zero_matrix()

    def _caculate_in_thread(self, *args, **kwargs):
        pred, label, ignore = args

        # # If not in streaming mode, clear matrix everytime when call `calculate`
        # if not self.streaming:
        #     self.zero_matrix()

        # label = np.transpose(label, (0, 2, 3, 1))
        # ignore = np.transpose(ignore, (0, 2, 3, 1))
        mask = np.array(ignore) == 1

        label = np.asarray(label)[mask]
        pred = np.asarray(pred)[mask]
        one = np.ones_like(pred)
        # Accumuate ([row=label, col=pred], 1) into sparse matrix
        spm = csr_matrix((one, (label, pred)),
                         shape=(self.num_classes, self.num_classes))
        spm = spm.todense()

        lock = kwargs['lock']
        lock.acquire()
        self.confusion_matrix += spm
        lock.release()

    def caculate(self, pred, label, ignore=None):
        AccuracyEvaluator.process_queue.put((self, (pred, label, ignore)))

    def zero_matrix(self):
        """ Clear confusion matrix """
        self.confusion_matrix = np.zeros([self.num_classes, self.num_classes],
                                         dtype='int64')

    def accuracy(self, *args, **kwargs):
        # 完成指标统计
        iou_list = []
        avg_iou = 0
        # TODO: use numpy sum axis api to simpliy
        vji = np.zeros(self.num_classes, dtype=int)
        vij = np.zeros(self.num_classes, dtype=int)
        for j in range(self.num_classes):
            v_j = 0
            for i in range(self.num_classes):
                v_j += self.confusion_matrix[j][i]
            vji[j] = v_j

        for i in range(self.num_classes):
            v_i = 0
            for j in range(self.num_classes):
                v_i += self.confusion_matrix[j][i]
            vij[i] = v_i

        for c in range(self.num_classes):
            total = vji[c] + vij[c] - self.confusion_matrix[c][c]
            if total == 0:
                iou = 0
            else:
                iou = float(self.confusion_matrix[c][c]) / total
            avg_iou += iou
            iou_list.append(iou)
        avg_iou = float(avg_iou) / float(self.num_classes)
        return avg_iou

    # def accuracy(self):
    #     total = self.confusion_matrix.sum()
    #     total_right = 0
    #     for c in range(self.num_classes):
    #         total_right += self.confusion_matrix[c][c]
    #     if total == 0:
    #         avg_acc = 0
    #     else:
    #         avg_acc = float(total_right) / total
    #
    #     vij = np.zeros(self.num_classes, dtype=int)
    #     for i in range(self.num_classes):
    #         v_i = 0
    #         for j in range(self.num_classes):
    #             v_i += self.confusion_matrix[j][i]
    #         vij[i] = v_i
    #
    #     acc_list = []
    #     for c in range(self.num_classes):
    #         if vij[c] == 0:
    #             acc = 0
    #         else:
    #             acc = self.confusion_matrix[c][c] / float(vij[c])
    #         acc_list.append(acc)
    #     return np.array(acc_list), avg_acc
    #
    # def kappa(self):
    #     vji = np.zeros(self.num_classes)
    #     vij = np.zeros(self.num_classes)
    #     for j in range(self.num_classes):
    #         v_j = 0
    #         for i in range(self.num_classes):
    #             v_j += self.confusion_matrix[j][i]
    #         vji[j] = v_j
    #
    #     for i in range(self.num_classes):
    #         v_i = 0
    #         for j in range(self.num_classes):
    #             v_i += self.confusion_matrix[j][i]
    #         vij[i] = v_i
    #
    #     total = self.confusion_matrix.sum()
    #
    #     # avoid spillovers
    #     # TODO: is it reasonable to hard code 10000.0?
    #     total = float(total) / 10000.0
    #     vji = vji / 10000.0
    #     vij = vij / 10000.0
    #
    #     tp = 0
    #     tc = 0
    #     for c in range(self.num_classes):
    #         tp += vji[c] * vij[c]
    #         tc += self.confusion_matrix[c][c]
    #
    #     tc = tc / 10000.0
    #     pe = tp / (total * total)
    #     po = tc / total
    #
    #     kappa = (po - pe) / (1 - pe)
    #     return kappa
