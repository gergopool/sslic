import torchvision.transforms as transforms
from typing import Callable

from .general import GeneralTransform
from .norm import normalize
from .utils import MultiCropTransform


class MocoTransform(GeneralTransform):

    def ssl(self, size: int, norm: str) -> Callable:
        aug = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize(norm)
        ])
        return MultiCropTransform([aug, aug])

    def train(self, size: int, norm) -> Callable:
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize(norm)
        ])

    def test(self, size: int, norm: str) -> Callable:
        return transforms.Compose([transforms.Resize(size), transforms.ToTensor(), normalize(norm)])

    def large(self, split: str = 'train', norm: str = 'imagenet') -> Callable:
        super().large(split, norm)
        if split in ['ssl', 'train']:
            return getattr(self, split)(224, norm)
        else:
            # test
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize(norm)
            ])

    def medium(self, split: str = 'train', norm: str = 'imagenet') -> Callable:
        super().medium(split, norm)
        return getattr(self, split)(64, norm)

    def small(self, split: str = 'train', norm: str = 'cifar10') -> Callable:
        super().medium(split, norm)
        return getattr(self, split)(32, norm)