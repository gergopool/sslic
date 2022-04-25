import torchvision.transforms as transforms
from typing import Callable
from torchvision.transforms.functional import InterpolationMode

from .norm import normalize
from .utils import MultiCropTransform
from .mocov2 import MocoTransform

__all__ = ['byol_transform']


class BYOLTransform(MocoTransform):

    def large(self, split: str = 'train', norm: str = 'imagenet') -> Callable:
        # Code from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        if split == 'ssl':
            aug_t = transforms.Compose([
                transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply(transforms.GaussianBlur(23, [.1, 2.]), p=0.1),
                transforms.RandomSolarize(128, p=0.2),
                transforms.ToTensor(),
                normalize(norm)
            ]),
            aug_s = transforms.Compose([
                transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply(transforms.GaussianBlur(23, [.1, 2.]), p=1.0),
                transforms.RandomSolarize(128, p=0.),
                transforms.ToTensor(),
                normalize(norm)
            ]),
            return MultiCropTransform([aug_t, aug_s])
        else:
            return super().large(split, norm)


def byol_transform():
    return BYOLTransform()