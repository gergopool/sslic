import torch
from torch import nn
import torch.nn.functional as F

from .general import Loss
from .mocov2 import Mocov2Loss

__all__ = ['nnclr_loss']

EPS = 1e-6


class NNCLRLoss(Mocov2Loss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = nn.CrossEntropyLoss()

    def cross_entropy(self, x, y):
        y = F.softmax(y, dim=1)
        x = F.log_softmax(x, dim=1)
        return torch.sum(-y * x, dim=1).mean()

    def get_nearest_neighbors(self, z: torch.Tensor, queue: torch.Tensor) -> torch.Tensor:
        sims = z @ queue.T
        return queue[sims.argmax(dim=1)]

    def sim_loss(self, nn: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        sim = p @ nn.T / self.tau
        labels = torch.arange(len(p), device=p.device)
        return self.criterion(sim, labels)

    def forward(self, p1: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor,
                z2: torch.Tensor) -> torch.Tensor:

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        p1 = F.normalize(p1, dim=1)
        p2 = F.normalize(p2, dim=1)

        queue = self.queue.clone().detach()
        nn1 = self.get_nearest_neighbors(z1, queue)
        nn2 = self.get_nearest_neighbors(z2, queue)

        loss = (self.sim_loss(nn1, p2) + self.sim_loss(nn2, p1)) * 0.5

        # Add the teacher embeddings to FIFO
        # The paper says it doesn't matter if both embeddings used or not
        # z1 is chosen because it's not guaranteed to be blurred.
        self.add_to_queue(z1)

        return loss

    @classmethod
    def imagenet(cls, *args, queue_len=98304, **kwargs):
        return cls(*args, queue_len=queue_len, **kwargs)

    @classmethod
    def tiny_imagenet(cls, *args, queue_len=16384, **kwargs):
        # Note: this queue_len was not defined in the paper
        return cls(*args, queue_len=queue_len, **kwargs)

    @classmethod
    def cifar10(cls, *args, queue_len=4096, **kwargs):
        # Note: this queue_len was not defined in the paper
        return cls(*args, queue_len=queue_len, **kwargs)

    @classmethod
    def cifar100(cls, *args, queue_len=4096, **kwargs):
        # Note: this queue_len was not defined in the paper
        return cls(*args, queue_len=queue_len, **kwargs)


def nnclr_loss() -> Loss:
    return NNCLRLoss