import torch
from torch import nn
import torch.nn.functional as F
import random

from ..ressl import ReSSLLoss
from ...utils import after_init_world_size_n_rank

__all__ = ['proto_ressl_loss', 'proto_ressl_cifar10_loss']

EPS = 1e-6


class ProtoReSSLLoss(ReSSLLoss):

    def __init__(self, *args, n_protos=4096, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_protos = n_protos
        w = torch.rand(n_protos, self.emb_dim) - 0.5
        w = F.normalize(w, dim=1)
        self.protos = nn.Parameter(w, requires_grad=False)
        self.queue = F.normalize(self.queue, dim=1)
        self.world_size, _ = after_init_world_size_n_rank()

    @property
    def queue_mean(self):
        return self.queue.mean(dim=0).detach().clone()

    @property
    def queue_std(self):
        return self.queue.std(dim=0).detach().clone()

    @property
    def queue_loss(self):
        queue = self.queue.clone().detach()
        protos = F.normalize(self.protos, dim=1)
        p = self.softmax(queue @ protos.T / self.tau_t)
        entropy = torch.sum(p.mean(dim=0) * torch.log(p.mean(dim=0) + EPS))
        return entropy / self.world_size

    def cross_entropy(self, x, y):
        return torch.sum(-y * torch.log(x + EPS), dim=1).mean()

    def forward(self, z_t: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:

        z_t = F.normalize(z_t, dim=1)  # Teacher, easy aug
        z_s = F.normalize(z_s, dim=1)  # Student, hard aug

        protos = F.normalize(self.protos, dim=1)  # Clusters

        p_s = self.softmax(z_s @ protos.T / self.tau_s)
        p_t = self.softmax(z_t @ protos.T / self.tau_t)

        # q = self.queue.clone().detach()
        # p_queue = self.softmax(z_t @ q.T / 0.1)  # batch_size x queue_len
        # p_t = self.softmax(p_queue @ q @ protos.T / self.tau_t)

        loss = self.cross_entropy(p_s, p_t) + self.queue_loss

        self.add_to_queue(z_t)

        return loss


def proto_ressl_loss(*args, **kwargs):
    return ProtoReSSLLoss(*args, **kwargs)


def proto_ressl_cifar10_loss(*args, **kwargs):
    return ProtoReSSLLoss(*args, emb_dim=128, n_protos=64, queue_len=4096, **kwargs)