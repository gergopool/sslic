import torch
from torch import nn


class Loss(nn.Module):

    def __init__(self, emb_dim: int = -1):
        super().__init__()
        self.emb_dim = emb_dim

    @classmethod
    def imagenet(cls, *args, **kwargs):
        return cls(*args, *kwargs)

    @classmethod
    def tiny_imagenet(cls, *args, **kwargs):
        return cls(*args, *kwargs)

    @classmethod
    def cifar10(cls, *args, **kwargs):
        return cls(*args, *kwargs)

    @classmethod
    def cifar100(cls, *args, **kwargs):
        return cls(*args, *kwargs)

    def step(self, progress: float):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
