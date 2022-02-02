import torch.nn.functional as F
import torch
from torch import nn

from ..utils import AllGather

__all__ = ['barlow_twins_loss']


class BarlowTwinsLoss(nn.Module):
    """Barlow Twins Loss

    Parameters
    ----------
    lambd : float, optional
        Lambda value to scale the off-diagonal values. 5e-3 by default.
    """

    def __init__(self, lambd: float = 5e-3):
        super(BarlowTwinsLoss, self).__init__()
        self.lambd = lambd

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:

        # Collect results from all gpus
        z1 = AllGather.apply(z1)
        z2 = AllGather.apply(z2)

        batch_size = len(z1)

        # Covariance matrix
        c = z1.T @ z2 / batch_size

        # Diag & Non-diag sums
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()

        # Loss
        loss = on_diag + self.lambd * off_diag

        return loss

    def off_diagonal(self, x: torch.Tensor) -> torch.Tensor:
        # Collect every value which is not on the diagonal
        n = len(x)
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_twins_loss() -> nn.Module:
    return BarlowTwinsLoss()