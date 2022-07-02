import torchvision.transforms as transforms
from typing import Callable, Tuple, List

from .general import GeneralTransform
from .norm import normalize
from .utils import MultiCropTransform

__all__ = ['mocov2_transform']


class MocoTransform(GeneralTransform):

    def __init__(self, blur_chance=0.5):
        super().__init__()
        self.blur_chance = blur_chance

    def multi_crop(self, sizes: List[int], scales: List[Tuple[float, float]],
                   norm: str) -> Callable:

        def base_trans(size: int, scale: int):
            kernel_size = int((size // 20) * 2) + 1
            aug = transforms.Compose([
                transforms.RandomResizedCrop(size, scale=scale),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size, [0.1, 2])],
                                       p=self.blur_chance),
                transforms.ToTensor(),
                normalize(norm)
            ])
            return aug

        trans = []
        for size, scale in zip(sizes, scales):
            trans.append(base_trans(size, scale))

        return MultiCropTransform(trans)

    def ssl(self, size: int, norm: str) -> Callable:
        kernel_size = int((size // 20) * 2) + 1
        aug = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size, [0.1, 2])],
                                   p=self.blur_chance),
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
        elif split == 'multi_crop':
            return self.multi_crop([224, 224, 192, 160, 128, 92],
                                   [(0.2, 1), (0.2, 1), (0.172, 0.86), (0.143, 0.715),
                                    (0.114, 0.571), (0.086, 0.429)],
                                   norm=norm)
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
        super().small(split, norm)
        return getattr(self, split)(32, norm)


def mocov2_transform():
    return MocoTransform()