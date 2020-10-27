# -*- coding: UTF-8 -*-
# @Time    : 2020/9/7 11:26 上午
# @File    : ImageNetV2.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import os
import torch.utils.data as data

from PIL import Image
import numpy as np


class ImageNetV2Data(data.Dataset):
    def __init__(self,
                 root,
                 image_set='train',
                 transform=None):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.image_set = image_set

        self.images = []
        self.labels = []
        if self.image_set == 'train':
            with open(os.path.join(root, 'train_list.txt'), 'r') as fp:
                content = fp.readline()
                while content:
                    kk = content.split(" ")
                    image_file = kk[0].strip()
                    label = kk[-1].replace('/n', '').strip()
                    if image_file != '' and label != '':
                        self.images.append(os.path.join(root, image_file))
                        self.labels.append(int(label))
                    else:
                        print('skip %s'%image_file)

                    content = fp.readline()
                    if content == '' or content == '\n':
                        break
        else:
            with open(os.path.join(root, 'val_list.txt'), 'r') as fp:
                content = fp.readline()
                while content:
                    kk = content.split(" ")
                    image_file = kk[0].strip()
                    label = kk[-1].replace('/n', '').strip()
                    if image_file != '' and label != '':
                        self.images.append(os.path.join(root, image_file))
                        self.labels.append(int(label))

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
        if self.transform is not None:
            img = self.transform(img)

        return img, self.labels[index]

    def __len__(self):
        return len(self.images)
