import torch
from torch import nn
import torch.nn.functional as F

from .general import Loss
from ..utils import AllGather

__all__ = ['mocov2_loss']

EPS = 1e-6


class Mocov2Loss(Loss):

    def __init__(self, *args, queue_len: int = 65536, tau: float = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_len = queue_len
        self.tau = tau

        self.register_buffer("queue", torch.randn(self.queue_len, self.emb_dim))
        self.queue = F.normalize(self.queue, dim=1)
        self.queue.requires_grad = False
        self.queue_idx = 0
        self.criterion = nn.CrossEntropyLoss()

    @classmethod
    def imagenet(cls, *args, queue_len=65536, **kwargs):
        return cls(*args, queue_len=queue_len, **kwargs)

    @classmethod
    def tiny_imagenet(cls, *args, queue_len=8192, **kwargs):
        return cls(*args, queue_len=queue_len, **kwargs)

    @classmethod
    def cifar10(cls, *args, queue_len=8192, **kwargs):
        return cls(*args, queue_len=queue_len, **kwargs)

    @classmethod
    def cifar100(cls, *args, queue_len=8192, **kwargs):
        return cls(*args, queue_len=queue_len, **kwargs)

    @torch.no_grad()
    def add_to_queue(self, z):
        z = AllGather.apply(z)
        assert self.queue_len % len(z) == 0
        self.queue[self.queue_idx:self.queue_idx + len(z)] = z.detach()
        self.queue_idx = (self.queue_idx + len(z)) % self.queue_len

    def forward(self, z_t: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:

        z_t = F.normalize(z_t, dim=1).detach()
        z_s = F.normalize(z_s, dim=1)
        labels = torch.zeros(len(z_t), dtype=torch.long).to(z_s.device, non_blocking=True)

        queue = self.queue.clone().detach()

        pos_samples = F.cosine_similarity(z_s, z_t, dim=1).view(len(z_s), 1)
        neg_samples = z_s @ queue.T

        samples = torch.cat((pos_samples, neg_samples), dim=1) / self.tau

        loss = self.criterion(samples, labels)

        self.add_to_queue(z_t)

        return loss


def mocov2_loss():
    return Mocov2Loss