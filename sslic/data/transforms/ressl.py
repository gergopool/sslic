import torchvision.transforms as transforms
from typing import Callable, Tuple, List

from .norm import normalize
from .utils import MultiCropTransform
from .mocov2 import MocoTransform

__all__ = ['ressl_transform']


class ReSSLTransform(MocoTransform):

    def large(self, split: str = 'train', norm: str = 'imagenet') -> Callable:
        if split == 'ssl':
            return MultiCropTransform([self._aug_t(224, norm), self._aug_s(224, norm, 0.5)])
        else:
            return super().large(split, norm)

    def medium(self, split: str = 'train', norm: str = 'imagenet') -> Callable:
        if split == 'ssl':
            return MultiCropTransform([self._aug_t(64, norm), self._aug_s(64, norm, 0.5)])
        else:
            return super().large(split, norm)

    def small(self, split: str = 'train', norm: str = 'cifar10') -> Callable:
        if split == 'ssl':
            return MultiCropTransform([self._aug_t(32, norm), self._aug_s(32, norm, 0.)])
        else:
            return super().large(split, norm)

    def multi_crop(self, sizes: List[int], scales: List[Tuple[float, float]],
                   norm: str) -> Callable:
        trans = [self._aug_t(sizes[0], scales[0], norm)]
        for size, scale in zip(sizes[1:], scales[1:]):
            trans.append(self._aug_s(size, scale, norm))
        return MultiCropTransform(trans)

    # ========================================================================
    # PRIVATE FUNCTIONS
    # ========================================================================

    def _aug_t(self,
               size: int,
               scale: Tuple[float, float] = (0.2, 1.),
               norm: str = 'imagenet') -> Callable:
        '''Teacher augmentations / weak augmentations'''
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=scale),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize(norm)
        ])

    def _aug_s(self,
               size: int,
               scale: Tuple[float, float] = (0.2, 1.),
               norm: str = 'imagenet',
               blur_chance: float = 0.5) -> Callable:
        '''Student augmentations / hard augmentations'''
        kernel_size = int((size // 20) * 2) + 1
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=scale),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size, [.1, 2.])], p=blur_chance),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize(norm)
        ])


def ressl_transform(*args, **kwargs):
    return ReSSLTransform(*args, **kwargs)