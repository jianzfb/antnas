import logging
import os

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import MNIST, CIFAR10, SVHN, CIFAR100
import numpy as np
from antnas.dataset.ext_transforms import *
from antnas.dataset.voc import VOCSegmentation
from antnas.dataset.portrait import PortraitSegmentation
from antnas.dataset.segmentation_dataset import *
from antnas.dataset.ImageNetV2 import *

logger = logging.getLogger(__name__)


class PlaceholderData(data.Dataset):
    def __init__(self, img_size, img_channels, out_channels):
        self.img_size = img_size
        self.img_channels = img_channels
        self.out_channels = out_channels
        if type(self.out_channels) == tuple:
            self.out_channels = self.out_channels[0]

    def __getitem__(self, index):
        random_img = \
            np.random.randint(0,
                              255,
                              (self.img_channels,self.img_size,self.img_size), dtype=np.uint8)
        random_norm_img = random_img/255.0
        random_norm_img = random_norm_img.astype(np.float32)
        return random_norm_img, (int)(np.random.random()*self.out_channels)

    def __len__(self):
        return 100


class PrefetchedWrapper(object):
    def prefetched_loader(loader):
        cuda_is_available = torch.cuda.is_available()
        if cuda_is_available:
            stream = torch.cuda.Stream()
            first = True
            mean = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1)

            input = None
            target = None
            for next_input, next_target in loader:
                with torch.cuda.stream(stream):
                    next_input = next_input.cuda(non_blocking=True)
                    next_target = next_target.cuda(non_blocking=True)
                    next_input = next_input.float()
                    next_input = next_input.sub_(mean).div_(std)

                if not first:
                    yield input, target
                else:
                    first = False

                torch.cuda.current_stream().wait_stream(stream)
                input = next_input
                target = next_target
            yield input, target
        else:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

            for next_input, next_target in loader:
                next_input = next_input.float()
                next_input = next_input.sub_(mean).div_(std)

                yield next_input, next_target

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.epoch = 0

    def __iter__(self):
        if (self.dataloader.sampler is not None and
            isinstance(self.dataloader.sampler,
                       torch.utils.data.distributed.DistributedSampler)):

            self.dataloader.sampler.set_epoch(self.epoch)
        self.epoch += 1
        return PrefetchedWrapper.prefetched_loader(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


class Test(data.Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        return np.random.randint(0, 255, (3, 512, 512), dtype=np.uint8), \
               np.random.randint(0, 20, (1, 512, 512), dtype=np.uint8)

    def __len__(self):
        return 100


def get_Test(path, *args):
    img_dim = 512
    in_channels = 3
    out_size = (512, 512)

    train_set = Test()
    val_set = Test()
    test_set = Test()

    return train_set, val_set, test_set, img_dim, in_channels, out_size


def get_PASCAL2012_SEG(path, *args):
    path = os.path.join(path, 'vision')
    img_dim = 224
    in_channels = 3
    out_size = (21, 224, 224)

    train_transform = ExtCompose([
        ExtRandomScale((0.5, 2.0)),
        ExtRandomCrop(size=(513, 513), pad_if_needed=True),
        ExtRandomHorizontalFlip(),
        ExtResize((224, 224)),
        ExtToTensor()
    ])

    val_transform = ExtCompose([
        ExtResize((224, 224)),
        ExtToTensor()
    ])

    train_dst = VOCSegmentation(root=path, is_aug=False, image_set='train',
                                transform=train_transform)

    val_dst = VOCSegmentation(root=path, is_aug=False, image_set='val',
                              transform=val_transform)
    test_dst = val_dst

    return train_dst, None, test_dst, img_dim, in_channels, out_size


def get_Portrait_SEG(path, *args):
    path = os.path.join(path, 'vision')
    img_dim = 512
    in_channels = 3
    out_size = (2, 512, 512)

    train_transform = ExtCompose([
        ExtRandomHorizontalFlip(),
        ExtResize((512, 512)),
        ExtToTensor(),
        ExtNormalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    val_transform = ExtCompose([
        ExtResize((512, 512)),
        ExtToTensor(),
        ExtNormalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    train_dst = PortraitSegmentation(root=path,
                                     image_set='train',
                                     transform=train_transform)

    val_dst = PortraitSegmentation(root=path,
                                   image_set='val',
                                   transform=val_transform)
    test_dst = val_dst

    return train_dst, None, test_dst, img_dim, in_channels, out_size


def get_SEG(path, *args):
    if os.path.exists(os.path.join(path, 'vision')):
        path = os.path.join(path, 'vision')

    img_dim = 384
    in_channels = 3
    out_size = (2, 384, 384)

    train_transform = ExtCompose([
        ExtRandomHorizontalFlip(),
        ExtResize((384, 384)),
        ExtToTensor(),
        ExtNormalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    val_transform = ExtCompose([
        ExtResize((384, 384)),
        ExtToTensor(),
        ExtNormalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    train_dst = SegmentationData(root=path,
                                     image_set='train',
                                     prefix='',
                                     transform=train_transform)

    val_dst = SegmentationData(root=path,
                                   image_set='val',
                                   prefix='',
                                   transform=val_transform)
    test_dst = val_dst

    return train_dst, None, test_dst, img_dim, in_channels, out_size


def get_ImageNetV2(path, *args):
    path = os.path.join(path, 'vision', 'ILSVRC2012_w')
    img_dim = 224
    in_channels = 3
    out_size = (1000,)

    # traindir = os.path.join(path, 'train')
    # valdir = os.path.join(path, 'validation')
    train_dst = ImageNetV2Data(root=path,
                               image_set='train',
                               transform=transforms.Compose([
                                transforms.RandomSizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()
                            ]))

    val_dst = ImageNetV2Data(root=path,
                             image_set='val',
                             transform=transforms.Compose([
                                transforms.Scale(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()
                            ]))

    test_dst = val_dst
    return train_dst, None, test_dst, img_dim, in_channels, out_size


def get_CIFAR10(path, *args):
    path = os.path.join(path, 'vision')
    img_dim = 32
    in_channels = 3
    out_size = (10,)
    val_size = 5000
    data_augmentation = [transforms.Pad(4),
                         transforms.RandomCrop(32),
                         transforms.RandomHorizontalFlip()]

    normalization = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    train_transfrom = transforms.Compose([
        transforms.Compose(data_augmentation),
        transforms.ToTensor(),
        normalization])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalization])

    train_set = CIFAR10(root=path, train=True, download=True, transform=train_transfrom)
    test_set = CIFAR10(root=path, train=False, download=True, transform=test_transform)
    # train_set, val_set = validation_split(train_set, train_transfrom, test_transform, val_size=val_size)

    return train_set, None, test_set, img_dim, in_channels, out_size


def get_CIFAR100(path, *args):
    path = os.path.join(path, 'vision')
    img_dim = 32
    in_channels = 3
    out_size = (100,)
    val_size = 5000

    data_augmentation = [transforms.Pad(4),
                         transforms.RandomCrop(32),
                         transforms.RandomHorizontalFlip()]

    train_transfrom = transforms.Compose([
        transforms.Compose(data_augmentation),
        transforms.ToTensor()])

    test_transform = transforms.Compose([
        transforms.ToTensor()])

    train_set = CIFAR100(root=path, train=True, download=True, transform=train_transfrom)
    test_set = CIFAR100(root=path, train=False, download=True, transform=test_transform)
    # train_set, val_set = validation_split(train_set, train_transfrom, test_transform, val_size=val_size)

    return train_set, None, test_set, img_dim, in_channels, out_size


def get_SVHN(path, *args):
    path = os.path.join(path, 'vision')
    img_dim = 32
    in_channels = 3
    out_size = (10,)
    val_size = 7500

    data_augmentation = [transforms.Pad(4),
                         transforms.RandomCrop(32)]

    normalization = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    train_transfrom = transforms.Compose([
        transforms.Compose(data_augmentation),
        transforms.ToTensor(),
        normalization])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalization])

    train_set = SVHN(root=path, split='train', download=True, transform=train_transfrom)
    test_set = SVHN(root=path, split='test', download=True, transform=test_transform)
    train_set, val_set = validation_split(train_set, train_transfrom, test_transform, val_size=val_size)

    return train_set, val_set, test_set, img_dim, in_channels, out_size


def get_MNIST(path, *args):
    path = os.path.join(path, 'vision')
    img_dim = 32
    in_channels = 1
    out_size = (10,)
    val_size = 10000

    data_augmentation = [transforms.Pad(4),
                         transforms.RandomCrop(32)]

    normalization = transforms.Normalize((0.1307,), (0.3081,))

    train_transfrom = transforms.Compose([
        transforms.Compose(data_augmentation),
        transforms.ToTensor(),
        normalization]
    )

    test_transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        normalization]
    )

    train_set = MNIST(root=path, train=True, download=True, transform=train_transfrom)
    test_set = MNIST(root=path, train=False, download=True, transform=test_transform)
    train_set, val_set = validation_split(train_set, train_transfrom, test_transform, val_size=val_size)

    return train_set, val_set, test_set, img_dim, in_channels, out_size


def get_ImageNet(path, *args):
    path = os.path.join(path, 'vision')
    img_dim = 224
    in_channels = 3
    out_size = (1000,)

    traindir = os.path.join(path, 'train')
    valdir = os.path.join(path, 'validation')

    train_set = datasets.ImageFolder(traindir, transforms.Compose([
        transforms.Scale(256),
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]))

    val_set = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]))

    return train_set, None, val_set, img_dim, in_channels, out_size


def get_Placeholder(path, kwargs):
    img_dim = kwargs['img_size']
    in_channels = kwargs['in_channels']
    out_size = (kwargs['out_channels'],)

    train_set = PlaceholderData(img_dim, in_channels, out_size)
    val_set = PlaceholderData(img_dim, in_channels, out_size)
    return train_set, None, val_set, img_dim, in_channels, out_size


sets = {
    'CIFAR10': get_CIFAR10,
    'CIFAR100': get_CIFAR100,
    'MNIST': get_MNIST,
    'SVHN': get_SVHN,
    'ImageNet': get_ImageNet,
    'ImageNetV2': get_ImageNetV2,
    'PASCAL2012SEG': get_PASCAL2012_SEG,
    'PLACEHOLDER': get_Placeholder,
}


def get_data(ds_name, batch_size, path, kwargs=None):
    logger.debug("Using {} dataset".format(ds_name))

    if ds_name in sets.keys():
        train_set, val_set, test_set, img_dim, in_channels, out_size = sets[ds_name](path, kwargs)
    else:
        raise ValueError("Dataset must in {}, got {}".format(sets.keys(), ds_name))

    logger.info("N train : %d" % len(train_set))
    logger.info("N test : %d" % len(test_set))

    train_loader = \
        DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=16, drop_last=True) if train_set is not None else None
    test_loader = \
        DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=8) if test_set is not None else None

    train_loader = PrefetchedWrapper(train_loader)
    test_loader = PrefetchedWrapper(test_loader)

    data_properties = {
        'img_dim': img_dim,
        'in_channels': in_channels,
        'out_size': out_size
    }

    return train_loader, None, test_loader, data_properties


class PartialDataset(torch.utils.data.Dataset):
    def __init__(self, parent_ds, offset, length, transform):
        self.parent_ds = parent_ds
        self.offset = offset
        self.length = length
        self.transform = transform
        assert len(parent_ds) >= offset + length, Exception("Parent Dataset not long enough")
        super(PartialDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        self.parent_ds.transform = self.transform
        return self.parent_ds[i + self.offset]


def validation_split(dataset, train_transforms, val_transforms, val_size=None, val_share=0.1):
    """
       Split a (training and validation combined) dataset into training and validation.
       Note that to be statistically sound, the items in the dataset should be statistically
       independent (e.g. not sorted by class, not several instances of the same dataset that
       could end up in either set).

       inputs:
          dataset:   ("training") dataset to split into training and validation
          val_share: fraction of validation dataset (should be 0<val_share<1, default: 0.1)
       returns: input dataset split into test_ds, val_ds

       """

    val_offset = len(dataset) - val_size if val_size is not None else int(len(dataset) * (1 - val_share))
    assert val_offset > 0, "Can't extract a size {} validation set out of a size {} dataset".format(val_size,
                                                                                                    len(dataset))
    return PartialDataset(dataset, 0, val_offset, train_transforms), PartialDataset(dataset, val_offset,
                                                                                    len(dataset) - val_offset,
                                                                                    val_transforms)
