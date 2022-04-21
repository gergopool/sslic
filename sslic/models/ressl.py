import torch
import torch.nn as nn
from torchvision import models
from typing import Callable
from copy import deepcopy

from ..losses import ressl_loss
from .base_model import BaseModel
from ..utils import after_init_world_size_n_rank, AllGather

__all__ = ['ressl_imagenet', 'ressl_tiny_imagenet', 'ressl_cifar10', 'ressl_cifar100']


class ReSSL(BaseModel):
    """
    ReSSL model
    Credits: https://github.com/KyleZheng1997/ReSSL/blob/main/network/ressl.py
    """

    def __init__(self,
                 base_encoder: nn.Module,
                 hidden_dim: int = 4096,
                 momentum=0.999,
                 ssl_loss: Callable = ressl_loss(),
                 **kwargs):
        super(ReSSL, self).__init__(base_encoder, ssl_loss=ssl_loss, **kwargs)
        self.hidden_dim = hidden_dim
        self.momentum = momentum
        self.world_size, self.rank = after_init_world_size_n_rank()

        # Projection head
        self.projector = nn.Sequential(nn.Linear(self.prev_dim, self.hidden_dim),
                                       nn.BatchNorm1d(self.hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(self.hidden_dim, self.dim))

        # Def networks
        self.student_net = nn.Sequential(self.encoder, self.projector)
        self.teacher_net = deepcopy(self.student_net)

        # Freeze
        for param in self.teacher_net.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _update_teacher(self):
        for s_param, t_param in zip(self.student_net.parameters(), self.teacher_net.parameters()):
            t_param.data = self.momentum * t_param.data + (1 - self.momentum) * s_param.data

    @torch.no_grad()
    def _batch_shuffle(self, x):
        # images from all gpus
        x = AllGather.apply(x)

        # random shuffle index
        idx_shuffle = torch.randperm(len(x)).to(x.device)
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        idx_this = idx_shuffle.view(self.world_size, -1)[self.rank]

        return x[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle(self, x, idx_unshuffle):
        # images from all gpus
        x = AllGather.apply(x)

        # idx in right order
        idx_this = idx_unshuffle.view(self.world_size, -1)[self.rank]

        return x[idx_this]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_easy, x_hard = x

        # Teacher side
        with torch.no_grad():
            self._update_teacher()

            if self.world_size > 1:
                shuffled_x, idx_unshuffle = self._batch_shuffle(x_easy)
                teacher_z = self.teacher_net(shuffled_x)
                teacher_z = self._batch_unshuffle(teacher_z, idx_unshuffle)
            else:
                teacher_z = self.teacher_net(x_easy)

        # Student side
        student_z = self.student_net(x_hard)

        return teacher_z, student_z


def ressl_imagenet(**kwargs) -> nn.Module:
    return ReSSL(models.resnet50,
                 dim=512,
                 hidden_dim=4096,
                 n_classes=1000,
                 zero_init_residual=True,
                 **kwargs)


def ressl_tiny_imagenet(**kwargs) -> nn.Module:
    from .cifar_resnet import resnet18
    return ReSSL(resnet18, dim=128, hidden_dim=512, momentum=0.996, n_classes=200, **kwargs)


def ressl_cifar10(**kwargs) -> nn.Module:
    from .cifar_resnet import resnet18
    return ReSSL(resnet18, dim=128, hidden_dim=512, momentum=0.99, n_classes=10, **kwargs)


def ressl_cifar100(**kwargs) -> nn.Module:
    from .cifar_resnet import resnet18
    return ReSSL(resnet18, dim=128, hidden_dim=512, momentum=0.99, n_classes=100, **kwargs)
