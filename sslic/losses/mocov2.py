import torch
from torch import nn
import torch.nn.functional as F

from ..utils import AllGather

__all__ = ['mocov2_loss', 'mocov2_tiny_imagenet_loss', 'mocov2_cifar10_loss']

EPS = 1e-6


class Mocov2Loss(nn.Module):

    def __init__(self, emb_dim=128, queue_len: int = 65536, tau: float = 0.07):
        super().__init__()
        self.emb_dim = emb_dim
        self.queue_len = queue_len
        self.tau = tau

        self.register_buffer("queue", torch.randn(self.queue_len, self.emb_dim))
        self.queue = F.normalize(self.queue, dim=1)
        self.queue.requires_grad = False
        self.queue_idx = 0
        self.criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def add_to_queue(self, z):
        z = AllGather.apply(z)
        assert self.queue_len % len(z) == 0
        self.queue[self.queue_idx:self.queue_idx + len(z)] = z.detach()
        self.queue_idx = (self.queue_idx + len(z)) % self.queue_len

    def forward(self, z_t: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:

        z_t = F.normalize(z_t, dim=1).detach()
        z_s = F.normalize(z_s, dim=1)

        queue = self.queue.clone().detach()

        pos_samples = F.cosine_similarity(z_s, z_t, dim=1).view(len(z_s), 1)
        neg_samples = z_s @ queue.T

        samples = torch.cat((pos_samples, neg_samples), dim=1) / self.tau
        labels = torch.zeros(len(samples)).to(z_s.device)

        loss = self.criterion(samples, labels)

        self.add_to_queue(z_t)

        return loss


def mocov2_loss(*args, **kwargs):
    return Mocov2Loss(*args, **kwargs)


def mocov2_tiny_imagenet_loss(*args, **kwargs):
    # Note: values created from ressl paper, it was not included in moco
    return Mocov2Loss(*args, queue_len=16384, **kwargs)


def mocov2_cifar10_loss(*args, **kwargs):
    # Note: values created from ressl paper, it was not included in moco
    return Mocov2Loss(*args, queue_len=4096, **kwargs)