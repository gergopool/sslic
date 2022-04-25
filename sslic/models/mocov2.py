import torch.nn as nn

from .momentum_model import MomentumModel

__all__ = ['mocov2_imagenet', 'mocov2_tiny_imagenet', 'mocov2_cifar10', 'mocov2_cifar100']
'''
Important note

Momentum parameteres were not reported for datasets other than Imagenet. Therefore,
for convinient comparisons, we maintain the same momentum parameters that ReSSL
uses for smaller datasets.
'''


def mocov2_tiny_imagenet(**kwargs) -> nn.Module:
    from .cifar_resnet import resnet18
    return MomentumModel(resnet18, dim=128, hidden_dim=128, momentum=0.996, n_classes=200, **kwargs)


def mocov2_cifar10(**kwargs) -> nn.Module:
    from .cifar_resnet import resnet18
    return MomentumModel(resnet18, dim=128, hidden_dim=128, momentum=0.99, n_classes=10, **kwargs)


def mocov2_cifar100(**kwargs) -> nn.Module:
    from .cifar_resnet import resnet18
    return MomentumModel(resnet18, dim=128, hidden_dim=128, momentum=0.99, n_classes=100, **kwargs)
