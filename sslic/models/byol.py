import torch
import torch.nn as nn
from torchvision import models
import math

from .momentum_model import MomentumModel

__all__ = ['byol_imagenet', 'byol_tiny_imagenet', 'byol_cifar10', 'byol_cifar100']
'''
Important note

Momentum parameteres were not reported for datasets other than Imagenet. Therefore,
for convinient comparisons, we maintain the same momentum parameters that ReSSL
uses for smaller datasets.
'''


class BYOL(MomentumModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictor = nn.Sequential(nn.Linear(self.dim, self.hidden_dim, bias=False),
                                       nn.BatchNorm1d(self.hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(self.hidden_dim, self.dim))
        self.base_momentum = self.momentum

    def _create_projector(self):
        # Add batchnorm to projection
        return nn.Sequential(nn.Linear(self.prev_dim, self.hidden_dim),
                             nn.BatchNorm1d(self.hidden_dim),
                             nn.ReLU(),
                             nn.Linear(self.hidden_dim, self.dim))

    def step(self, progress: float):
        scale = 0.5 * (1. + math.cos(math.pi * progress))
        self.momentum = scale + self.base_momentum * (1 - scale)

    def _student_forward(self, x):
        return self.predictor(self.student_net(x))

    @torch.no_grad()
    def _teacher_forward(self, x):
        # No batch shuffle needed
        return self.teacher_net(x)


def byol_imagenet(**kwargs) -> nn.Module:
    return MomentumModel(models.resnet50,
                         dim=256,
                         hidden_dim=4096,
                         momentum=0.99,
                         n_classes=1000,
                         zero_init_residual=True,
                         **kwargs)


def byol_tiny_imagenet(**kwargs) -> nn.Module:
    from .cifar_resnet import resnet18
    return MomentumModel(resnet18, dim=256, hidden_dim=4096, momentum=0.99, n_classes=200, **kwargs)


def byol_cifar10(**kwargs) -> nn.Module:
    from .cifar_resnet import resnet18
    return MomentumModel(resnet18, dim=256, hidden_dim=4096, momentum=0.99, n_classes=10, **kwargs)


def byol_cifar100(**kwargs) -> nn.Module:
    from .cifar_resnet import resnet18
    return MomentumModel(resnet18,
                         dim=256,
                         hidden_dim=4096,
                         pred_hidden_dim=4096,
                         momentum=0.99,
                         n_classes=100,
                         **kwargs)
