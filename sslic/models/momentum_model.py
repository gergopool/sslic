import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy
from typing import List, Tuple

from .split_batch_norm import SplitBatchNorm2d
from .base_model import BaseModel
from ..utils import after_init_world_size_n_rank, AllGather


class MomentumModel(BaseModel):
    """
    General Momentum model
    Credits: https://github.com/facebookresearch/moco/blob/main/moco/builder.py
    """

    def __init__(self, *args, hidden_dim: int = 4096, momentum=0.999, **kwargs):
        super(MomentumModel, self).__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.momentum = momentum
        self.world_size, self.rank = after_init_world_size_n_rank()

        # Projection head
        self.projector = self._create_projector()

        # Def networks
        self.student_net = nn.Sequential(self.encoder, self.projector)
        self.teacher_net = deepcopy(self.student_net)

        # Freeze
        for param in self.teacher_net.parameters():
            param.requires_grad = False

    def _create_projector(self):
        '''Projector. It is made here, so children classes can change it'''
        return nn.Sequential(nn.Linear(self.prev_dim, self.hidden_dim),
                             nn.ReLU(),
                             nn.Linear(self.hidden_dim, self.dim))

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

    def _student_forward(self, x):
        return self.student_net(x)

    @torch.no_grad()
    def _teacher_forward(self, x):
        if self.world_size > 1:
            shuffled_x, idx_unshuffle = self._batch_shuffle(x)
            z = self.teacher_net(shuffled_x)
            z = self._batch_unshuffle(z, idx_unshuffle)
        else:
            z = self.teacher_net(x)
        return z

    def forward(self, xs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_easy = xs[0]
        x_hard = xs[1:]

        # Teacher side
        with torch.no_grad():
            self._update_teacher()
            teacher_z = self._teacher_forward(x_easy)

        # Student side
        student_z = []
        for x in x_hard:
            student_z.append(self._student_forward(x))
        student_z = torch.cat(student_z, dim=0)

        return teacher_z, student_z

    @classmethod
    def imagenet(cls, *args, **kwargs) -> BaseModel:
        if after_init_world_size_n_rank()[0] == 1:
            kwargs["norm_layer"] = partial(SplitBatchNorm2d, num_splits=8)
            kwargs['sync_batchnorm'] = False
        return super().imagenet(*args, **kwargs)

    @classmethod
    def tiny_imagenet(cls, *args, **kwargs) -> BaseModel:
        if after_init_world_size_n_rank()[0] == 1:
            kwargs["norm_layer"] = partial(SplitBatchNorm2d, num_splits=8)
            kwargs['sync_batchnorm'] = False
        return super().tiny_imagenet(*args, **kwargs)

    @classmethod
    def cifar10(cls, *args, **kwargs) -> BaseModel:
        if after_init_world_size_n_rank()[0] == 1:
            kwargs["norm_layer"] = partial(SplitBatchNorm2d, num_splits=8)
            kwargs['sync_batchnorm'] = False
        return super().cifar10(*args, **kwargs)

    @classmethod
    def cifar100(cls, *args, **kwargs) -> BaseModel:
        if after_init_world_size_n_rank()[0] == 1:
            kwargs["norm_layer"] = partial(SplitBatchNorm2d, num_splits=8)
            kwargs['sync_batchnorm'] = False
        return super().cifar100(*args, **kwargs)