import torch
from torch import nn
import torch.nn.functional as F

from ..utils import AllGather

__all__ = ['ressl_loss', 'ressl_tiny_imagenet_loss', 'ressl_cifar10_loss']

EPS = 1e-6


class ReSSLLoss(nn.Module):

    def __init__(self,
                 emb_dim=512,
                 queue_len: int = 65536,
                 tau_s: float = 0.1,
                 tau_t: float = 0.04):
        super().__init__()
        self.emb_dim = emb_dim
        self.queue_len = queue_len
        self.tau_s = tau_s
        self.tau_t = tau_t

        self.register_buffer("queue", torch.randn(self.queue_len, self.emb_dim))
        self.queue = F.normalize(self.queue, dim=1)
        self.queue.requires_grad = False
        self.queue_idx = 0
        self.softmax = nn.Softmax(dim=1)

    def cross_entropy(self, x, y):
        return torch.sum(-y * torch.log(x + EPS), dim=1).mean()

    @torch.no_grad()
    def add_to_queue(self, z):
        z = AllGather.apply(z)
        assert self.queue_len % len(z) == 0
        self.queue[self.queue_idx:self.queue_idx + len(z)] = z.detach()
        self.queue_idx = (self.queue_idx + len(z)) % self.queue_len

    def forward(self, z_t: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:

        z_t = F.normalize(z_t, dim=1)
        z_s = F.normalize(z_s, dim=1)

        queue = self.queue.clone().detach()

        p_s = self.softmax(z_s @ queue.T / self.tau_s)
        p_t = self.softmax(z_t @ queue.T / self.tau_t)

        loss = self.cross_entropy(p_s, p_t)

        self.add_to_queue(z_t)

        return loss


def ressl_loss(*args, **kwargs):
    return ReSSLLoss(*args, **kwargs)


def ressl_tiny_imagenet_loss(*args, **kwargs):
    return ReSSLLoss(128, *args, queue_len=16384, **kwargs)


def ressl_cifar10_loss(*args, **kwargs):
    return ReSSLLoss(128, *args, queue_len=4096, tau_t=0.05, **kwargs)