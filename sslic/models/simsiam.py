import torch
import torch.nn as nn
from typing import Tuple, List

from ..losses.simsiam import SimSiamLoss
from .base_model import BaseModel

__all__ = ['simsiam_model']


class SimSiam(BaseModel):
    """
    Build a SimSiam model.
    Credits: https://github.com/facebookresearch/simsiam/

    Parameters
        ----------
        pred_dim : int, optional
            Dimension of bottleneck layer at predict layer, by default 512
    """
    default_loss = SimSiamLoss

    def __init__(self, *args, pred_dim: int = 512, mlp_len=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.pred_dim = pred_dim

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
        self.projector = nn.Sequential(*projector_layers)

        # Predictor
        self.predictor = nn.Sequential(nn.Linear(self.dim, self.pred_dim, bias=False),
                                       nn.BatchNorm1d(self.pred_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(self.pred_dim, self.dim))

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        x1, x2 = xs[:2]
        small_crops = xs[2:]

        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        ps = []
        for crop in small_crops:
            ps.append(self.predictor(self.projector(self.encoder(crop))))

        ps = torch.cat(ps, dim=0) if ps else torch.Tensor(0)

        return p1, p2, z1, z2, ps

    @classmethod
    def imagenet(cls, *args, **kwargs) -> BaseModel:
        kwargs.setdefault("dim", 2048)
        kwargs.setdefault("pred_dim", 512)
        kwargs.setdefault("mlp_len", 3)
        return super().imagenet(*args, **kwargs)

    @classmethod
    def tiny_imagenet(cls, *args, **kwargs) -> BaseModel:
        kwargs.setdefault("dim", 2048)
        kwargs.setdefault("pred_dim", 512)
        kwargs.setdefault("mlp_len", 2)
        return super().tiny_imagenet(*args, **kwargs)

    @classmethod
    def cifar10(cls, *args, **kwargs) -> BaseModel:
        kwargs.setdefault("dim", 2048)
        kwargs.setdefault("pred_dim", 512)
        kwargs.setdefault("mlp_len", 2)
        return super().cifar10(*args, **kwargs)

    @classmethod
    def cifar100(cls, *args, **kwargs) -> BaseModel:
        kwargs.setdefault("dim", 2048)
        kwargs.setdefault("pred_dim", 512)
        kwargs.setdefault("mlp_len", 2)
        return super().cifar100(*args, **kwargs)


class MultiviewSimsiam(SimSiam):

    def forward(self, xs: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:

        results = []
        for x in xs:
            z = self.projector(self.encoder(x))
            p = self.predictor(z)
            results.append((p, z))

        return tuple(results)


def simsiam_model() -> SimSiam:
    return SimSiam


def multiview_simsiam_model() -> SimSiam:
    return MultiviewSimsiam
