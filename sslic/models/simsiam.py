import torch
import torch.nn as nn
from typing import Callable

from ..losses import simsiam_loss
from .base_model import BaseModel
from torchvision import models

__all__ = ['simsiam_imagenet', 'simsiam_tiny_imagenet', 'simsiam_cifar10', 'simsiam_cifar100']


class SimSiam(BaseModel):
    """
    Build a SimSiam model.
    Credits: https://github.com/facebookresearch/simsiam/

    Parameters
        ----------
        pred_dim : int, optional
            Dimension of bottleneck layer at predict layer, by default 512
    """

    def __init__(self,
                 base_encoder: nn.Module,
                 pred_dim: int = 512,
                 mlp_len=3,
                 ssl_loss: Callable = simsiam_loss(),
                 **kwargs):
        super(SimSiam, self).__init__(base_encoder, ssl_loss=ssl_loss, **kwargs)
        self.pred_dim = pred_dim

        # Projector
        projector_layers = []
        for _ in range(mlp_len - 1):
            projector_layers.extend([
                nn.Linear(self.prev_dim, self.prev_dim, bias=False),
                nn.BatchNorm1d(self.prev_dim),
                nn.ReLU(inplace=True)
            ])
        projector_layers.extend([nn.Linear(self.prev_dim, self.dim, bias=True)])
        self.projector = nn.Sequential(*projector_layers)

        self.standardize = nn.BatchNorm1d(self.dim, affine=False)

        # Predictor
        self.pred_dim = self.dim
        self.predictor = nn.Sequential(
            #    nn.Linear(self.dim, self.pred_dim, bias=False),
            #    nn.BatchNorm1d(self.pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pred_dim, self.dim))
        # for param in self.predictor.parameters():
        #     param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x

        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        z1 = self.projector(h1)
        z2 = self.projector(h2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, p2, self.standardize(z1), self.standardize(z2)


def simsiam_imagenet(**kwargs) -> nn.Module:
    return SimSiam(models.resnet50,
                   pred_dim=512,
                   mlp_len=3,
                   dim=2048,
                   n_classes=1000,
                   zero_init_residual=True,
                   **kwargs)


def simsiam_tiny_imagenet(**kwargs) -> nn.Module:
    from .cifar_resnet import resnet18
    return SimSiam(resnet18, pred_dim=512, mlp_len=2, dim=2048, n_classes=200, **kwargs)


def simsiam_cifar10(**kwargs) -> nn.Module:
    from .cifar_resnet import resnet18
    return SimSiam(resnet18, pred_dim=512, mlp_len=2, dim=2048, n_classes=10, **kwargs)


def simsiam_cifar100(**kwargs) -> nn.Module:
    from .cifar_resnet import resnet18
    return SimSiam(resnet18, pred_dim=512, mlp_len=2, dim=2048, n_classes=100, **kwargs)
