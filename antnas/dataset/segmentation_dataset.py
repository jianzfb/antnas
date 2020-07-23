# -*- coding: UTF-8 -*-
# @Time    : 2020/7/21 6:36 下午
# @File    : segmentation_dataset.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import os
import torch.utils.data as data

from PIL import Image
import numpy as np


class SegmentationData(data.Dataset):
    def __init__(self,
                 root,
                 image_set='train',
                 prefix='',
                 transform=None):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.image_set = image_set

        self.images = []
        self.masks = []
        if self.image_set == 'train':
            with open(os.path.join(root, 'train.txt'), 'r') as fp:
                content = fp.readline()
                while content:
                    content = content.replace('/n', '')
                    kk = content.split(" ")
                    image_file = kk[0].strip()
                    mask_file = kk[-1].strip()
                    if image_file != '' and mask_file != '':
                        self.images.append(os.path.join(root, prefix, image_file))
                        self.masks.append(os.path.join(root, prefix, mask_file))
                    else:
                        print('skip %s'%image_file)

                    content = fp.readline()
                    if content == '' or content == '\n':
                        break
        else:
            with open(os.path.join(root, 'val.txt'), 'r') as fp:
                content = fp.readline()
                while content:
                    content = content.replace('/n', '')
                    kk = content.split(" ")
                    image_file = kk[0].strip()
                    mask_file = kk[-1].strip()
                    if image_file != '' and mask_file != '':
                        self.images.append(os.path.join(root, prefix, image_file))
                        self.masks.append(os.path.join(root, prefix, mask_file))

                    content = fp.readline()
                    if content == '' or content == '\n':
                        break

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        target = np.array(target)
        target = Image.fromarray(target)

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)
