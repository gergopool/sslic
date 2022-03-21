import torch
import torch.nn as nn
from torchvision import models

from ..losses.simclr import simclr_loss
from .base_model import BaseModel

__all__ = ['simclr_imagenet', 'simclr_cifar10', 'simclr_cifar100']


class SimCLR(BaseModel):
    """
    SimCLR model
    Credits: https://github.com/google-research/simclr
    """

    def __init__(self, base_encoder: nn.Module, **kwargs):
        super(SimCLR, self).__init__(base_encoder, ssl_loss=simclr_loss(), **kwargs)

        # Projection head
        self.projector = nn.Sequential(nn.Linear(self.prev_dim, self.prev_dim),
                                       nn.BatchNorm1d(self.prev_dim),
                                       nn.ReLU(),
                                       nn.Linear(self.prev_dim, self.dim, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x

        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        z1 = self.projector(h1)
        z2 = self.projector(h2)

        return h1, (z1, z2)


def simclr_imagenet() -> nn.Module:
    return SimCLR(models.resnet50, dim=512, n_classes=1000, zero_init_residual=True)


def simclr_cifar10() -> nn.Module:
    from .cifar_resnet import resnet18
    return SimCLR(resnet18, dim=128, n_classes=10)


def simclr_cifar100() -> nn.Module:
    from .cifar_resnet import resnet18
    return SimCLR(resnet18, dim=128, n_classes=100)
