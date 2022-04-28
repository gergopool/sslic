import torch
import torch.nn as nn

from ..losses.simclr import InfoNCE
from .base_model import BaseModel

__all__ = ['simclr_model']


class SimCLR(BaseModel):
    """
    SimCLR model
    Credits: https://github.com/google-research/simclr
    """

    default_loss = InfoNCE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        return z1, z2

    @classmethod
    def imagenet(cls, *args, **kwargs) -> BaseModel:
        return super().imagenet(*args, dim=512, **kwargs)

    @classmethod
    def tiny_imagenet(cls, *args, **kwargs) -> BaseModel:
        return super().tiny_imagenet(*args, dim=128, **kwargs)

    @classmethod
    def cifar10(cls, *args, **kwargs) -> BaseModel:
        return super().cifar10(*args, dim=128, **kwargs)

    @classmethod
    def cifar100(cls, *args, **kwargs) -> BaseModel:
        return super().cifar100(*args, dim=128, **kwargs)


def simclr_model() -> SimCLR:
    return SimCLR
