import torch.nn as nn

from ..losses import info_nce_loss
from .base_model import BaseModel
from torchvision import models

__all__ = ['simclr_imagenet', 'simclr_cifar10', 'simclr_cifar100']

class SimCLR(BaseModel):
    """
    Build a SimCLR model.
    """
    def __init__(self, base_encoder, **kwargs):
        super(SimCLR, self).__init__(base_encoder, ssl_loss=info_nce_loss, **kwargs)

        # This part is based upon the official google simclr code.
        # https://github.com/google-research/simclr/blob/master/tf2/model.py
        # Projection head
        self.projector = nn.Sequential(nn.Linear(self.prev_dim, self.prev_dim),
                                       nn.BatchNorm1d(self.prev_dim),
                                       nn.ReLU(),
                                       nn.Linear(self.prev_dim, self.dim, bias=False))

    def forward(self, x):
        x1, x2 = x

        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        z1 = self.projector(h1)
        z2 = self.projector(h2)

        y_hat = self.classifier(h1.detach())

        return h1.detach(), (z1, z2)

def simclr_imagenet(dim=2048):
    return SimCLR(models.resnet50, dim=dim, n_classes=1000, zero_init_residual=True)

def simclr_cifar10(dim=128, **kwargs):
    from .cifar_resnet import resnet18
    return SimCLR(resnet18, dim=dim, n_classes=10)

def simclr_cifar100(dim=128, **kwargs):
    from .cifar_resnet import resnet18
    return SimCLR(resnet18, dim=dim, n_classes=100)
