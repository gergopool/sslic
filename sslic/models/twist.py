import torch
import torch.nn as nn

from ..losses.twist import TwistLoss
from .base_model import BaseModel

__all__ = ['twist_model']


class Twist(BaseModel):
    """
    Build a twist model.
    Credits: https://github.com/bytedance/TWIST

    Parameters
        ----------
        hidden_dim : int, optional
            Dimension of hidden layers in projection head
    """

    default_loss = TwistLoss

    def __init__(self, *args, hidden_dim: int = 4096, mlp_len=3, **kwargs):
        super(Twist, self).__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim

        # Projector
        projector_layers = []
        for _ in range(mlp_len - 1):
            projector_layers.extend([
                nn.Linear(self.prev_dim, self.prev_dim, bias=False),
                nn.BatchNorm1d(self.prev_dim),
                nn.ReLU(inplace=True)
            ])
        projector_layers.extend([
            nn.Linear(self.prev_dim, self.dim, bias=False), nn.BatchNorm1d(self.dim, affine=False)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x

        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        z1 = self.projector(h1)
        z2 = self.projector(h2)

        return z1, z2

    @classmethod
    def imagenet(cls, *args, **kwargs) -> BaseModel:
        return super().imagenet(*args, dim=4096, mlp_len=3, hidden_dim=4096, **kwargs)

    @classmethod
    def tiny_imagenet(cls, *args, **kwargs) -> BaseModel:
        return super().tiny_imagenet(*args, dim=512, mlp_len=2, pred_dim=512, **kwargs)

    @classmethod
    def cifar10(cls, *args, **kwargs) -> BaseModel:
        return super().tiny_imagenet(*args, dim=512, mlp_len=2, pred_dim=512, **kwargs)

    @classmethod
    def cifar100(cls, *args, **kwargs) -> BaseModel:
        return super().tiny_imagenet(*args, dim=512, mlp_len=2, pred_dim=512, **kwargs)


def twist_model() -> Twist:
    return Twist
