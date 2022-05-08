import torch
import torch.nn.functional as F
from torch import nn

from .general import Loss

__all__ = ['linear_eval_loss']


class LinearEvalLoss(Loss):
    # This is only created to follow the design
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.criterion(x, y)


def linear_eval_loss() -> Loss:
    return LinearEvalLoss
