import torch
import torch.nn as nn
import math

from ..losses.byol import BYOLLoss
from .momentum_model import MomentumModel

__all__ = ['byol_model']
'''
Important note

Momentum parameteres were not reported for datasets other than Imagenet. Therefore,
for convinient comparisons, we maintain the same momentum parameters that ReSSL
uses for smaller datasets.
'''


class BYOL(MomentumModel):

    default_loss = BYOLLoss

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictor = nn.Sequential(nn.Linear(self.dim, self.hidden_dim, bias=False),
                                       nn.BatchNorm1d(self.hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(self.hidden_dim, self.dim))
        self.base_momentum = self.momentum

    def _create_projector(self):
        # Add batchnorm to projection
        return nn.Sequential(nn.Linear(self.prev_dim, self.hidden_dim),
                             nn.BatchNorm1d(self.hidden_dim),
                             nn.ReLU(),
                             nn.Linear(self.hidden_dim, self.dim))

    def step(self, progress: float):
        scale = 0.5 * (1. + math.cos(math.pi * progress))
        self.momentum = scale + self.base_momentum * (1 - scale)

    def _student_forward(self, x):
        return self.predictor(self.student_net(x))

    @torch.no_grad()
    def _teacher_forward(self, x):
        # No batch shuffle needed
        return self.teacher_net(x)

    @classmethod
    def imagenet(cls, *args, **kwargs) -> MomentumModel:
        return super().imagenet(*args, dim=256, hidden_dim=4096, momentum=0.996, **kwargs)

    @classmethod
    def tiny_imagenet(cls, *args, **kwargs) -> MomentumModel:
        return super().tiny_imagenet(*args, dim=256, hidden_dim=4096, momentum=0.996, **kwargs)

    @classmethod
    def cifar10(cls, *args, **kwargs) -> MomentumModel:
        return super().tiny_imagenet(*args, dim=256, hidden_dim=4096, momentum=0.996, **kwargs)

    @classmethod
    def cifar100(cls, *args, **kwargs) -> MomentumModel:
        return super().tiny_imagenet(*args, dim=256, hidden_dim=4096, momentum=0.996, **kwargs)


def byol_model() -> BYOL:
    return BYOL