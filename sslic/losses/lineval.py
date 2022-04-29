import torch
from torch import nn

from .general import Loss

__all__ = ['lineval_loss']


class LinEvalLoss(Loss):
    """SimSiam Loss"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.criterion(x, y)


def lineval_loss() -> Loss:
    return LinEvalLoss
