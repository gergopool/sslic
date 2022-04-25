import torch.nn as nn
from torchvision import models

from .momentum_model import MomentumModel

__all__ = ['ressl_imagenet', 'ressl_tiny_imagenet', 'ressl_cifar10', 'ressl_cifar100']


def ressl_imagenet(**kwargs) -> nn.Module:
    return MomentumModel(models.resnet50,
                         dim=512,
                         hidden_dim=4096,
                         momentum=0.999,
                         n_classes=1000,
                         zero_init_residual=True,
                         **kwargs)


def ressl_tiny_imagenet(**kwargs) -> nn.Module:
    from .cifar_resnet import resnet18
    return MomentumModel(resnet18, dim=128, hidden_dim=512, momentum=0.996, n_classes=200, **kwargs)


def ressl_cifar10(**kwargs) -> nn.Module:
    from .cifar_resnet import resnet18
    return MomentumModel(resnet18, dim=128, hidden_dim=512, momentum=0.99, n_classes=10, **kwargs)


def ressl_cifar100(**kwargs) -> nn.Module:
    from .cifar_resnet import resnet18
    return MomentumModel(resnet18, dim=128, hidden_dim=512, momentum=0.99, n_classes=100, **kwargs)
