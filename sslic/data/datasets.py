import torchvision.datasets as datasets
import torch
import os
from . import transforms


def get_dataset_provider(root, method_name, dataset_name):
    method_name = method_name.lower()
    if method_name in ['simclr', 'simsiam']:
        return MocoDataset(root, dataset_name)
    elif method_name == 'barlow_twins':
        return BarlowTwinsDataset(root, dataset_name)
    else:
        raise NameError(f"Method unknown: {method_name.lower()}")


class MocoDataset:
    def __init__(self, root, dataset_name='imagenet'):
        self.root = root
        self.dataset_name = dataset_name.lower()
        if self.dataset_name not in ['imagenet', 'cifar10', 'cifar100']:
            raise NameError(f"Unknown dataset_name: {self.dataset_name}")
        self.dataset_fn = getattr(self, self.dataset_name)

    @property
    def n_classes(self):
        return {"imagenet": 1000, "cifar10": 10, "cifar100": 100}[self.dataset_name]

    def __call__(self, split='ssl') -> torch.utils.data.Dataset:
        is_train = split in ['ssl', 'train']
        return self.dataset_fn(split, is_train)

    def imagenet(self, split, is_train):
        imagenet_dir = "train" if is_train else "val"
        imagenet_dir = os.path.join(self.root, imagenet_dir)
        trans = transforms.imagenet_mocov2(split)
        return datasets.ImageFolder(imagenet_dir, trans)

    def cifar10(self, split, is_train):
        trans = transforms.small_moco_like("cifar10", split=split)
        return datasets.CIFAR10(self.root, is_train, trans)

    def cifar100(self, split, is_train):
        trans = transforms.small_moco_like("cifar100", split=split)
        return datasets.CIFAR10(self.root, is_train, trans)


class BarlowTwinsDataset(MocoDataset):
    def imagenet(self, split, is_train):
        imagenet_dir = "train" if is_train else "val"
        imagenet_dir = os.path.join(self.root, imagenet_dir)
        trans = transforms.imagenet_barlow_twins(split)
        return datasets.ImageFolder(imagenet_dir, trans)


if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    from transforms import NORMS

    # Retrive img
    root = "/data/shared/data/imagenet"
    dataset = get_dataset_provider(root, "simsiam", "imagenet")("ssl")
    x, y = dataset[0]

    # Convert to human readable image
    img = x[0].detach().permute(1, 2, 0).cpu().numpy()
    img = img * np.array(NORMS['imagenet']['std']) \
              + np.array(NORMS['imagenet']['mean'])
    img = np.round(img * 255).clip(0, 255).astype(np.uint8)

    # Save
    img = Image.fromarray(img)
    img.save("test.jpg")
