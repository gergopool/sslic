import torch
import torch.nn as nn

from ..losses.barlow_twins import BarlowTwinsLoss
from .base_model import BaseModel

__all__ = ['barlow_twins_model']


class BarlowTwins(BaseModel):
    """
    Barlow Twins model.
    Credits: https://github.com/facebookresearch/barlowtwins/
    """

    default_loss = BarlowTwinsLoss

    def __init__(self, *args, **kwargs):
        super(BarlowTwins, self).__init__(*args, **kwargs)

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

        return z1, z2

    @classmethod
    def imagenet(cls, *args, **kwargs) -> BaseModel:
        return super().imagenet(*args, dim=8096, **kwargs)

    @classmethod
    def tiny_imagenet(cls, *args, **kwargs) -> BaseModel:
        return super().tiny_imagenet(*args, dim=512, **kwargs)

    @classmethod
    def cifar10(cls, *args, **kwargs) -> BaseModel:
        return super().cifar10(*args, dim=512, **kwargs)

    @classmethod
    def cifar100(cls, *args, **kwargs) -> BaseModel:
        return super().cifar100(*args, dim=512, **kwargs)


def barlow_twins_model() -> BarlowTwins:
    return BarlowTwins