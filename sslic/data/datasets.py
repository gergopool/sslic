import torchvision.datasets as datasets
import torch
import os
from .transforms import get_transform_generator

__all__ = ["simsiam_datasets", "simclr_datasets", "barlow_twins_datasets", "ressl_datasets"]


class DatasetGenerator:

    def __init__(self, root, transform_generator, dataset_name='imagenet'):
        self.root = root
        self.transform_generator = transform_generator
        self.dataset_name = dataset_name.lower()
        if self.dataset_name not in ['imagenet', 'cifar10', 'cifar100', 'tiny_imagenet']:
            raise NameError(f"Unknown dataset_name: {self.dataset_name}")
        self.dataset_fn = getattr(self, self.dataset_name)

    def __call__(self, split='ssl') -> torch.utils.data.Dataset:
        is_train = split in ['ssl', 'train']
        return self.dataset_fn(split, is_train)

    def imagenet(self, split, is_train):
        imagenet_dir = "train" if is_train else "val"
        imagenet_dir = os.path.join(self.root, imagenet_dir)
        trans = self.transform_generator.make_by_dataset('imagenet', split)
        return datasets.ImageFolder(imagenet_dir, trans)

    def tiny_imagenet(self, split, is_train):
        imagenet_dir = "train" if is_train else "val"
        imagenet_dir = os.path.join(self.root, imagenet_dir)
        trans = self.transform_generator.make_by_dataset('tiny_imagenet', split)
        return datasets.ImageFolder(imagenet_dir, trans)

    def cifar10(self, split, is_train):
        trans = self.transform_generator.make_by_dataset("cifar10", split=split)
        return datasets.CIFAR10(self.root, is_train, trans)

    def cifar100(self, split, is_train):
        trans = self.transform_generator.make_by_dataset("cifar100", split=split)
        return datasets.CIFAR100(self.root, is_train, trans)


def simsiam_datasets(root, dataset_name):
    return DatasetGenerator(root, get_transform_generator('simsiam'), dataset_name)


def simclr_datasets(root, dataset_name):
    return DatasetGenerator(root, get_transform_generator('simclr'), dataset_name)


def barlow_twins_datasets(root, dataset_name):
    return DatasetGenerator(root, get_transform_generator('barlow_twins'), dataset_name)


def ressl_datasets(root, dataset_name):
    return DatasetGenerator(root, get_transform_generator('ressl'), dataset_name)


if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    from transforms.norm import NORMS

    # Retrive img
    root = "/data/shared/data/tiny_imagenet"
    dataset = ressl_datasets(root, "tiny_imagenet")("ssl")
    x, y = dataset[0]

    # Convert to human readable image
    img = x[0].detach().permute(1, 2, 0).cpu().numpy()
    img = img * np.array(NORMS['imagenet']['std']) \
              + np.array(NORMS['imagenet']['mean'])
    img = np.round(img * 255).clip(0, 255).astype(np.uint8)

    # Save
    img = Image.fromarray(img)
    img.save("test.jpg")
