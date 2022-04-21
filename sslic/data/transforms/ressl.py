import torchvision.transforms as transforms
from typing import Callable

from .norm import normalize
from .utils import GaussianBlur, MultiCropTransform
from .moco import MocoTransform


class ReSSLTransform(MocoTransform):

    def aug_t(self, size: int, norm: str = 'imagenet') -> Callable:
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize(norm)
        ])

    def aug_s(self, size: int, norm: str = 'imagenet', blur_chance: float = 0.5) -> Callable:
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=blur_chance),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize(norm)
        ])

    def large(self, split: str = 'train', norm: str = 'imagenet') -> Callable:
        if split == 'ssl':
            return MultiCropTransform([self.aug_t(224, norm), self.aug_s(224, norm, 0.5)])
        else:
            return super().large(split, norm)

    def medium(self, split: str = 'train', norm: str = 'imagenet') -> Callable:
        if split == 'ssl':
            return MultiCropTransform([self.aug_t(64, norm), self.aug_s(64, norm, 0.5)])
        else:
            return super().large(split, norm)

    def small(self, split: str = 'train', norm: str = 'cifar10') -> Callable:
        if split == 'ssl':
            return MultiCropTransform([self.aug_t(32, norm), self.aug_s(32, norm, 0.)])
        else:
            return super().large(split, norm)