import torchvision.transforms as transforms
from typing import Callable
from torchvision.transforms.functional import InterpolationMode

from .norm import normalize
from .utils import GaussianBlur, Solarization, MultiCropTransform
from .moco import MocoTransform


class BarlowTwinsTransform(MocoTransform):

    def large(self, split: str = 'train', norm: str = 'imagenet') -> Callable:
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
                normalize(norm)
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
                normalize(norm)
            ])
            return MultiCropTransform([aug1, aug2])
        else:
            return super().large(split, norm)
