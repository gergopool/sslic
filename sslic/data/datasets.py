import torchvision.datasets as datasets
import torch
import os
from . import transforms


def get_dataset_provider(root, dataset_name, method_name=None):
    func_name = method_name + "_datasets"
    if func_name not in globals():
        raise NameError(f"Self-supervised dataset {method_name} is unknown.")
    return globals()[func_name](root, dataset_name)


class DatasetGenerator:

    def __init__(self, root, dataset_name='imagenet', imagenet_trans=None, cifar_trans=None):
        self.root = root
        self.dataset_name = dataset_name.lower()
        if self.dataset_name not in ['imagenet', 'cifar10', 'cifar100']:
            raise NameError(f"Unknown dataset_name: {self.dataset_name}")
        self.dataset_fn = getattr(self, self.dataset_name)
        self.imagenet_trans = imagenet_trans
        self.cifar_trans = cifar_trans

    def __call__(self, split='ssl') -> torch.utils.data.Dataset:
        is_train = split in ['ssl', 'train']
        return self.dataset_fn(split, is_train)

    def imagenet(self, split, is_train):
        imagenet_dir = "train" if is_train else "val"
        imagenet_dir = os.path.join(self.root, imagenet_dir)
        trans = self.imagenet_trans(split)
        return datasets.ImageFolder(imagenet_dir, trans)

    def cifar10(self, split, is_train):
        trans = self.cifar_trans("cifar10", split=split)
        return datasets.CIFAR10(self.root, is_train, trans)

    def cifar100(self, split, is_train):
        trans = self.cifar_transe("cifar100", split=split)
        return datasets.CIFAR100(self.root, is_train, trans)


def simsiam_datasets(root, dataset_name):
    return DatasetGenerator(root,
                            dataset_name,
                            transforms.imagenet_mocov2,
                            transforms.small_moco_like)


def simclr_datasets(root, dataset_name):
    return DatasetGenerator(root,
                            dataset_name,
                            transforms.imagenet_mocov2,
                            transforms.small_moco_like)


def barlow_twins_datasets(root, dataset_name):
    return DatasetGenerator(root,
                            dataset_name,
                            transforms.imagenet_barlow_twins,
                            transforms.small_moco_like)


def ressl_datasets(root, dataset_name):
    return DatasetGenerator(root, dataset_name, transforms.imagenet_ressl, transforms.small_ressl)


if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    from transforms import NORMS

    # Retrive img
    root = "/data/shared/data/imagenet"
    dataset = get_dataset_provider(root, "imagenet", method_name="ressl")("ssl")
    x, y = dataset[0]

    # Convert to human readable image
    img = x[0].detach().permute(1, 2, 0).cpu().numpy()
    img = img * np.array(NORMS['imagenet']['std']) \
              + np.array(NORMS['imagenet']['mean'])
    img = np.round(img * 255).clip(0, 255).astype(np.uint8)

    # Save
    img = Image.fromarray(img)
    img.save("test.jpg")
