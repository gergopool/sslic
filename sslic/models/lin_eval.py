import torch
import torch.nn as nn

from ..losses.lin_eval import LinearEvalLoss
from .base_model import BaseModel

__all__ = ['lin_eval_model']


class LinearEvalModel(BaseModel):
    """
    Linear Evaluation Model
    """

    default_loss = LinearEvalLoss

    def __init__(self, *args, n_classes=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes

        # Linear classifier head
        self.classifier = nn.Linear(self.prev_dim, self.n_classes)

        # Freeze params
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Reinit classifier
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.classifier.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.encoder(x)
        x = self.classifier(x.float())
        return x

    @classmethod
    def imagenet(cls, *args, n_classes=1000, **kwargs) -> BaseModel:
        return super().imagenet(*args, n_classes=n_classes, **kwargs)

    @classmethod
    def tiny_imagenet(cls, *args, n_classes=200, **kwargs) -> BaseModel:
        return super().tiny_imagenet(*args, n_classes=n_classes, **kwargs)

    @classmethod
    def cifar10(cls, *args, n_classes=10, **kwargs) -> BaseModel:
        return super().cifar10(*args, n_classes=n_classes, **kwargs)

    @classmethod
    def cifar100(cls, *args, n_classes=100, **kwargs) -> BaseModel:
        return super().cifar100(*args, n_classes=n_classes, **kwargs)


def lin_eval_model() -> LinearEvalModel:
    return LinearEvalModel
