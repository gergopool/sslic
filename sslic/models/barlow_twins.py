import torch
import torch.nn as nn

from ..losses.barlow_twins import barlow_twins_loss
from .base_model import BaseModel
from torchvision import models

__all__ = ['barlow_twins_imagenet', 'barlow_twins_cifar10', 'barlow_twins_cifar100']


class BarlowTwins(BaseModel):
    """
    Barlow Twins model.
    Credits: https://github.com/facebookresearch/barlowtwins/
    """

    def __init__(self, *args, **kwargs):
        super(BarlowTwins, self).__init__(*args, ssl_loss=barlow_twins_loss(), **kwargs)

        # This part is based upon the official code linked in the paper.
        #
        # Projection head
        self.projector = nn.Sequential(nn.Linear(self.prev_dim, self.dim, bias=False),
                                       nn.BatchNorm1d(self.dim),
                                       nn.ReLU(),
                                       nn.Linear(self.dim, self.dim, bias=False),
                                       nn.BatchNorm1d(self.dim),
                                       nn.ReLU(),
                                       nn.Linear(self.dim, self.dim, bias=False),
                                       nn.BatchNorm1d(self.dim, affine=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x

        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        z1 = self.projector(h1)
        z2 = self.projector(h2)

        return h1.detach(), (z1, z2)


def barlow_twins_imagenet() -> nn.Module:
    return BarlowTwins(models.resnet50, dim=8096, n_classes=1000, zero_init_residual=True)


def barlow_twins_cifar10() -> nn.Module:
    from .cifar_resnet import resnet18
    return BarlowTwins(resnet18, dim=512, n_classes=10)


def barlow_twins_cifar100() -> nn.Module:
    from .cifar_resnet import resnet18
    return BarlowTwins(resnet18, dim=512, n_classes=100)