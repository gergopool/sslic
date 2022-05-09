from abc import ABC
from typing import Callable

__all__ = ['']


class GeneralTransform(ABC):

    datasets = ['imagenet', 'tiny_imagenet', 'cifar10', 'cifar100']

    def make_by_dataset(self, dataset_name: str, split: str = 'train') -> Callable:

        assert split in ['ssl', 'multi_crop', 'train', 'test'], f"Unknown split: {split}"

        if dataset_name == 'imagenet':
            return self.large(split, norm='imagenet')
        elif dataset_name == 'tiny_imagenet':
            return self.medium(split, norm='tiny_imagenet')
        elif dataset_name == 'cifar10':
            return self.small(split, norm='cifar10')
        elif dataset_name == 'cifar100':
            return self.small(split, norm='cifar100')
        else:
            raise NameError(f"Unknown dataset: {dataset_name}")

    def large(self, split: str = 'train', norm: str = 'imagenet') -> Callable:
        assert split in ['ssl', 'multi_crop', 'train', 'test'], f"Unknown split: {split}"

    def medium(self, split: str = 'train', norm: str = 'imagenet') -> Callable:
        assert split in ['ssl', 'train', 'test'], f"Unknown split: {split}"

    def small(self, split: str = 'train', norm: str = 'cifar10') -> Callable:
        assert split in ['ssl', 'train', 'test'], f"Unknown split: {split}"