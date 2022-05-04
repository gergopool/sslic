import torch
import torch.nn.functional as F
from torch import nn

from .general import Loss
from ..utils import AllGather

__all__ = ['simclr_loss']


class InfoNCE(Loss):
    """Info NCE loss (used in SimCLR)

        Parameters
        ----------
        tau : float, optional
            The value controlling the scale of cosine similarities,
            by default 0.07.
        """

    def __init__(self, *args, tau: float = 0.07, **kwargs):

        super().__init__(*args, **kwargs)
        self.tau = tau

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:

        # Collect from all gpu
        z1 = AllGather.apply(z1)
        z2 = AllGather.apply(z2)

        # Combine views and normalize
        z = torch.cat((z1, z2), dim=0)
        z = F.normalize(z, dim=1)
        n = len(z)

        # Labels telling which images make pairs
        ones = torch.ones(n // 2).to(z.device, non_blocking=True)
        labels = ones.diagflat(n // 2) + ones.diagflat(-n // 2)

        # Note: The following code might require a large amount of memory
        # in case of large batch size
        sim_m = z @ z.T

        # This is a bit of cheat. Instead of removing cells from
        # the matrix where i==j, instead we set it to a very small value
        sim_m = sim_m.fill_diagonal_(-1) / self.tau

        # Get probability distribution
        sim_m = torch.nn.functional.log_softmax(sim_m, dim=1)

        # Choose values on which we calculate the loss
        loss = -torch.sum(sim_m * labels) / n

        return loss


def simclr_loss() -> Loss:
    return InfoNCE