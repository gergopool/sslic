import torch.nn as nn

from ..losses import barlow_twins_loss
from .base_model import BaseModel
from torchvision import models

__all__ = ['barlow_twins_imagenet', 'barlow_twins_cifar10', 'barlow_twins_cifar100']

class BarlowTwins(BaseModel):
    """
    Build a Barlow Twins model.
    """
    def __init__(self, base_encoder, **kwargs):
        super(BarlowTwins, self).__init__(base_encoder, ssl_loss=barlow_twins_loss, **kwargs)

        # This part is based upon the official code linked in the paper.
        # https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # Projection head
        self.projector = nn.Sequential(nn.Linear(self.prev_dim, self.dim, bias=False),
                                       nn.BatchNorm1d(self.dim),
                                       nn.ReLU(),
                                       nn.Linear(self.dim, self.dim, bias=False),
                                       nn.BatchNorm1d(self.dim),
                                       nn.ReLU(),
                                       nn.Linear(self.dim, self.dim, bias=False),
                                       nn.BatchNorm1d(self.dim, affine=False))

    def forward(self, x):
        x1, x2 = x

        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        z1 = self.projector(h1)
        z2 = self.projector(h2)

        y_hat = self.classifier(h1.detach())

        return y_hat, (z1, z2)

def barlow_twins_imagenet(dim=8096, **kwargs):
    return BarlowTwins(models.resnet50, dim=dim, n_classes=1000)

def barlow_twins_cifar10(dim=512, **kwargs):
    return BarlowTwins(models.resnet18, dim=dim, n_classes=10)

def barlow_twins_cifar100(dim=512, **kwargs):
    return BarlowTwins(models.resnet18, dim=dim, n_classes=100)