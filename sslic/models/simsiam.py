import torch.nn as nn

from ..losses import simsiam_loss
from .base_model import BaseModel
from torchvision import models

__all__ = ['simsiam_imagenet', 'simsiam_cifar10', 'simsiam_cifar100']


class SimSiam(BaseModel):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, pred_dim=512, **kwargs):
        super(SimSiam, self).__init__(base_encoder, ssl_loss=simsiam_loss, **kwargs)
        self.pred_dim = pred_dim

        # Projector
        self.projector = nn.Sequential(nn.Linear(self.prev_dim, self.prev_dim, bias=False),
                                       nn.BatchNorm1d(self.prev_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(self.prev_dim, self.prev_dim, bias=False),
                                       nn.BatchNorm1d(self.prev_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(self.prev_dim, self.dim, bias=False),
                                       nn.BatchNorm1d(self.dim, affine=False))

        # Predictor
        self.predictor = nn.Sequential(nn.Linear(self.dim, self.pred_dim, bias=False),
                                       nn.BatchNorm1d(self.pred_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(self.pred_dim, self.dim))

    def forward(self, x):
        x1, x2 = x

        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        z1 = self.projector(h1)
        z2 = self.projector(h2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        y_hat = self.classifier(h1.detach())

        return y_hat, (p1, p2, z1.detach(), z2.detach())


def simsiam_imagenet(pred_dim=512, dim=2048, **kwargs):
    return SimSiam(models.resnet50, pred_dim=pred_dim, dim=dim, n_classes=1000)

def simsiam_cifar10(pred_dim=32, dim=128, **kwargs):
    return SimSiam(models.resnet18, pred_dim=pred_dim, dim=dim, n_classes=10)

def simsiam_cifar100(pred_dim=32, dim=128, **kwargs):
    return SimSiam(models.resnet18, pred_dim=pred_dim, dim=dim, n_classes=100)


