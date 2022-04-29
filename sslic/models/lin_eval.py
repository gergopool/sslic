import torch
import torch.nn as nn

from ..losses.lineval import LinEvalLoss
from .base_model import BaseModel

__all__ = ['lineval_model']


class LinEval(BaseModel):

    default_loss = LinEvalLoss

    def __init__(self, *args, n_classes=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
        self.classifier = nn.Sequential(nn.Linear(self.prev_dim, self.n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            embs = self.encoder(x)

        return self.classifier(embs)

    @classmethod
    def imagenet(cls, *args, **kwargs) -> BaseModel:
        return super().imagenet(*args, n_classes=1000, **kwargs)

    @classmethod
    def tiny_imagenet(cls, *args, **kwargs) -> BaseModel:
        return super().tiny_imagenet(*args, n_classes=200, **kwargs)

    @classmethod
    def cifar10(cls, *args, **kwargs) -> BaseModel:
        return super().cifar10(*args, n_classes=10, **kwargs)

    @classmethod
    def cifar100(cls, *args, **kwargs) -> BaseModel:
        return super().cifar100(*args, n_classes=100, **kwargs)


def lineval_model() -> LinEval:
    return LinEval
