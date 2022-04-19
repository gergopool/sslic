import torchvision.transforms as transforms
import random
import torch
from PIL import Image, ImageFilter, ImageOps
from torchvision.transforms import Normalize
from torchvision.transforms.functional import InterpolationMode

# =======================================================================
# Normalization
# =======================================================================

NORMS = {
    "imagenet": {
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]
    },
    "cifar10": {
        "mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010]
    },
    "cifar100": {
        "mean": [0.5071, 0.4867, 0.4408], "std": [0.2675, 0.2565, 0.2761]
    }
}


def normalize(dataset_name):
    '''Get normalization transformation of dataset'''
    if dataset_name not in NORMS:
        raise NameError(f"No norm values have been defined to dataset {dataset_name}")
    norm_data = NORMS[dataset_name]
    return Normalize(**norm_data)


# =======================================================================
# Method specific augmentation
# =======================================================================


def imagenet_mocov2(split='train'):
    if split == 'ssl':
        aug = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize("imagenet")
        ])
        return MultiCropTransform([aug, aug])
    elif split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize("imagenet")
        ])
    elif split == 'test':
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize("imagenet")
        ])
    else:
        raise NameError(f"Split name unknown: {split}")


def small_moco_like(dataset_name='cifar10', split='train'):
    if split == 'ssl':
        aug = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize(dataset_name)
        ])
        return MultiCropTransform([aug, aug])
    elif split == 'train':
        return transforms.Compose([
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize(dataset_name)
        ])
    elif split == 'test':
        return transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor(), normalize(dataset_name)])
    else:
        raise NameError(f"Split name unknown: {split}")


def imagenet_barlow_twins(split='train'):
    # Code from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    if split == 'ssl':
        aug1 = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur([.1, 2.]),
            Solarization(p=0.0),
            transforms.ToTensor(),
            normalize("imagenet")
        ])
        aug2 = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur([.1, 2.]),
            Solarization(p=0.2),
            transforms.ToTensor(),
            normalize("imagenet")
        ])
        return MultiCropTransform([aug1, aug2])
    else:
        return imagenet_mocov2(split)


def imagenet_ressl(split='train'):
    if split == 'ssl':
        aug_t = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize("imagenet")
        ])
        aug_s = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize("imagenet")
        ])
        return MultiCropTransform([aug_t, aug_s])
    else:
        return imagenet_mocov2(split)


def small_ressl(dataset_name='cifar10', split='train'):
    if split == 'ssl':
        aug_t = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize(dataset_name)
        ])
        aug_s = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize(dataset_name)
        ])
        return MultiCropTransform([aug_t, aug_s])
    else:
        return small_moco_like(dataset_name, split)


class Solarization(object):

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


# =======================================================================
# Multi-crop strategies
# =======================================================================


class MultiCropTransform:

    def __init__(self, trans_list):
        self.trans_list = trans_list

    def __call__(self, x):
        x_out = []
        for trans in self.trans_list:
            x_out.append(trans(x))
        return x_out