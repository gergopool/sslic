import torchvision.datasets as datasets
import os
from .transforms import get_transform

__all__ = ["imagenet_dataset", "tiny_imagenet_dataset", "cifar10_dataset", "cifar100_dataset"]


def _is_train(split):
    return split in ['ssl', 'multi_crop', 'train']


def imagenet_dataset(root: str, method_name: str, split: str, norm='imagenet'):
    imagenet_dir = "train" if _is_train(split) else "val"
    imagenet_dir = os.path.join(root, imagenet_dir)
    trans = get_transform(method_name, "imagenet", split, norm=norm)
    return datasets.ImageFolder(imagenet_dir, trans)


def tiny_imagenet_dataset(root: str, method_name: str, split: str, norm='tiny_imagenet'):
    imagenet_dir = "train" if _is_train(split) else "val"
    imagenet_dir = os.path.join(root, imagenet_dir)
    trans = get_transform(method_name, "tiny_imagenet", split, norm=norm)
    return datasets.ImageFolder(imagenet_dir, trans)


def cifar10_dataset(root: str, method_name: str, split: str, norm='cifar10'):
    trans = get_transform(method_name, "cifar10", split, norm=norm)
    return datasets.CIFAR10(root, _is_train(split), trans)


def cifar100_dataset(root: str, method_name: str, split: str, norm='cifar100'):
    trans = get_transform(method_name, "cifar100", split, norm=norm)
    return datasets.CIFAR100(root, _is_train(split), trans)


if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    from transforms.norm import NORMS

    # Retrive img
    root = "/data/shared/data/tiny_imagenet"
    dataset = tiny_imagenet_dataset(root, "byol", "ssl")
    x, y = dataset[0]

    # Convert to human readable image
    img = x[0].detach().permute(1, 2, 0).cpu().numpy()
    img = img * np.array(NORMS['imagenet']['std']) \
              + np.array(NORMS['imagenet']['mean'])
    img = np.round(img * 255).clip(0, 255).astype(np.uint8)

    # Save
    img = Image.fromarray(img)
    img.save("test.jpg")
