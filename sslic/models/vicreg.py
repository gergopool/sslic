import torch
import torch.nn as nn
from typing import Callable

from ..losses import vicreg_loss
from .base_model import BaseModel
from torchvision import models

__all__ = ['vicreg_imagenet', 'vicreg_tiny_imagenet', 'vicreg_cifar10', 'vicreg_cifar100']


class VICReg(BaseModel):
    """
    Build a vicreg model.
    Credits: https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py

    Parameters
        ----------
        pred_dim : int, optional
            Dimension of bottleneck layer at predict layer, by default 512
    """

    def __init__(self,
                 base_encoder: nn.Module,
                 mlp_len=3,
                 ssl_loss: Callable = vicreg_loss(),
                 **kwargs):
        super().__init__(base_encoder, ssl_loss=ssl_loss, **kwargs)

        # Projector
        projector_layers = [nn.Linear(self.prev_dim, self.dim, bias=False)]
        for _ in range(mlp_len - 1):
            projector_layers.extend([
                nn.BatchNorm1d(self.dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.dim, self.dim, bias=False)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x

        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        z1 = self.projector(h1)
        z2 = self.projector(h2)

        return z1, z2


def vicreg_imagenet(**kwargs) -> nn.Module:
    return VICReg(models.resnet50,
                  mlp_len=3,
                  dim=8192,
                  n_classes=1000,
                  zero_init_residual=True,
                  **kwargs)


def vicreg_tiny_imagenet(**kwargs) -> nn.Module:
    from .cifar_resnet import resnet18
    # Note: This is undeclared in paper and therefore using custom metrics
    return VICReg(resnet18, mlp_len=2, dim=1024, n_classes=200, **kwargs)


def vicreg_cifar10(**kwargs) -> nn.Module:
    from .cifar_resnet import resnet18
    return VICReg(resnet18, mlp_len=2, dim=1024, n_classes=10, **kwargs)


def vicreg_cifar100(**kwargs) -> nn.Module:
    from .cifar_resnet import resnet18
    return VICReg(resnet18, mlp_len=2, dim=1024, n_classes=100, **kwargs)