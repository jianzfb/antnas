# @Time    : 2019-09-17 14:49
# @Author  : zhangchenming
import os
import torch.utils.data as data

from PIL import Image
import numpy as np


class PortraitSegmentation(data.Dataset):
    def __init__(self,
                 root,
                 image_set='train',
                 transform=None):

        self.root = os.path.expanduser(root)
        self.transform = transform

        self.image_set = image_set

        self.portrait_root = os.path.join(self.root, image_set)

        if not os.path.isdir(self.portrait_root):
            raise RuntimeError('Dataset not found or corrupted')

        split_f = os.path.join(self.root, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError('no image list')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(self.portrait_root, x) for x in file_names]
        self.masks = [os.path.join(self.portrait_root, x[0:x.rfind('/') - 5] + 'label' + x[x.rfind('/'): -3] + 'png')
                      for x in file_names]
        assert (len(self.images) == len(self.masks))

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
        target = target / 255
        target = Image.fromarray(target)

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)
