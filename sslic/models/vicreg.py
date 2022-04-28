import torch
import torch.nn as nn

from ..losses.vicreg import VICRegLoss
from .base_model import BaseModel

__all__ = ['vicreg_model']


class VICReg(BaseModel):

    default_loss = VICRegLoss
    """
    Build a vicreg model.
    Credits: https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py

    Parameters
        ----------
        pred_dim : int, optional
            Dimension of bottleneck layer at predict layer, by default 512
    """

    def __init__(self, *args, mlp_len=3, **kwargs):
        super().__init__(*args, **kwargs)

        # Projector
        projector_layers = [nn.Linear(self.prev_dim, self.dim, bias=False)]
        for _ in range(mlp_len - 1):
            projector_layers.extend([
                nn.BatchNorm1d(self.dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.dim, self.dim, bias=False)
            ])
        self.projector = nn.Sequential(*projector_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x

        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        z1 = self.projector(h1)
        z2 = self.projector(h2)

        return z1, z2

    @classmethod
    def imagenet(cls, *args, **kwargs) -> BaseModel:
        return super().imagenet(*args, dim=8192, mlp_len=3, **kwargs)

    @classmethod
    def tiny_imagenet(cls, *args, **kwargs) -> BaseModel:
        # Note: This is undeclared in paper and therefore using custom metrics
        return super().tiny_imagenet(*args, dim=1024, mlp_len=2, **kwargs)

    @classmethod
    def cifar10(cls, *args, **kwargs) -> BaseModel:
        # Note: This is undeclared in paper and therefore using custom metrics
        return super().cifar10(*args, dim=1024, mlp_len=2, **kwargs)

    @classmethod
    def cifar100(cls, *args, **kwargs) -> BaseModel:
        # Note: This is undeclared in paper and therefore using custom metrics
        return super().cifar100(*args, dim=1024, mlp_len=2, **kwargs)


def vicreg_model() -> VICReg:
    return VICReg