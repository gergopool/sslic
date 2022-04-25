import torch
import torch.nn as nn
from typing import Callable

from ..losses import twist_loss
from .base_model import BaseModel
from torchvision import models

__all__ = ['twist_imagenet', 'twist_tiny_imagenet', 'twist_cifar10', 'twist_cifar100']


class Twist(BaseModel):
    """
    Build a twist model.
    Credits: https://github.com/bytedance/TWIST

    Parameters
        ----------
        hidden_dim : int, optional
            Dimension of hidden layers in projection head
    """

    def __init__(self,
                 base_encoder: nn.Module,
                 hidden_dim: int = 4096,
                 mlp_len=3,
                 ssl_loss: Callable = twist_loss(),
                 **kwargs):
        super(Twist, self).__init__(base_encoder, ssl_loss=ssl_loss, **kwargs)
        self.hidden_dim = hidden_dim

        # Projector
        projector_layers = []
        for _ in range(mlp_len - 1):
            projector_layers.extend([
                nn.Linear(self.prev_dim, self.prev_dim, bias=False),
                nn.BatchNorm1d(self.prev_dim),
                nn.ReLU(inplace=True)
            ])
        projector_layers.extend([nn.Linear(self.prev_dim, self.dim, bias=True)])

        self.standardize = nn.BatchNorm1d(self.dim, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x

        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        z1 = self.standardize(self.projector(h1))
        z2 = self.standardize(self.projector(h2))

        return z1, z2


def twist_imagenet(**kwargs) -> nn.Module:
    return Twist(models.resnet50,
                 hidden_dim=4096,
                 mlp_len=3,
                 dim=4096,
                 n_classes=1000,
                 zero_init_residual=True,
                 **kwargs)


def twist_tiny_imagenet(**kwargs) -> nn.Module:
    from .cifar_resnet import resnet18
    # Note: This is undeclared in paper and therefore using custom metrics
    return Twist(resnet18, hidden_dim=512, mlp_len=2, dim=512, n_classes=200, **kwargs)


def twist_cifar10(**kwargs) -> nn.Module:
    from .cifar_resnet import resnet18
    # Note: This is undeclared in paper and therefore using custom metrics
    return Twist(resnet18, hidden_dim=512, mlp_len=2, dim=512, n_classes=10, **kwargs)


def twist_cifar100(**kwargs) -> nn.Module:
    from .cifar_resnet import resnet18
    # Note: This is undeclared in paper and therefore using custom metrics
    return Twist(resnet18, hidden_dim=512, mlp_len=2, dim=512, n_classes=100, **kwargs)
