import torch
import torch.nn.functional as F

from .general import Loss
from .mocov2 import Mocov2Loss

__all__ = ['ressl_loss']

EPS = 1e-6


class ReSSLLoss(Mocov2Loss):

    def __init__(self, *args, tau_s=0.1, tau_t=0.04, **kwargs):
        super().__init__(*args, tau=tau_s, **kwargs)
        self.tau_s = self.tau  # for convenient naming
        self.tau_t = tau_t

    @classmethod
    def imagenet(cls, *args, queue_len=131072, **kwargs):
        return cls(*args, queue_len=queue_len, **kwargs)

    @classmethod
    def tiny_imagenet(cls, *args, queue_len=16384, **kwargs):
        return cls(*args, queue_len=queue_len, **kwargs)

    @classmethod
    def cifar10(cls, *args, queue_len=16384, **kwargs):
        return cls(*args, queue_len=queue_len, **kwargs)

    @classmethod
    def cifar100(cls, *args, queue_len=16384, **kwargs):
        return cls(*args, queue_len=queue_len, **kwargs)

    def cross_entropy(self, x, y):
        return torch.sum(-y * torch.log(x + EPS), dim=1).mean()

    def forward(self, z_t: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:

        # Normalize the embedings
        z_t = F.normalize(z_t, dim=1)
        z_s = F.normalize(z_s, dim=1)

        # Queue is always normalized, no need for normalization
        queue = self.queue.clone().detach()

        # Calculate scaled similarities
        p_s = F.softmax(z_s @ queue.T / self.tau_s, dim=1)
        p_t = F.softmax(z_t @ queue.T / self.tau_t, dim=1)
        loss = self.cross_entropy(p_s, p_t)

        # Add the teacher embeddings to FIFO
        self.add_to_queue(z_t)

        return loss


def ressl_loss() -> Loss:
    return ReSSLLoss