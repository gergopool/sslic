import torch
import torch.nn.functional as F
from torch import nn

from ...utils import AllGather

__all__ = ['ts_simsiam_loss']

EPS = 1e-6


class TSSimSiamLoss(nn.Module):
    """TSSimSiam Loss"""

    def __init__(self, tau: float = 0.1, sharpen_temp: float = 0.25):
        super(TSSimSiamLoss, self).__init__()
        self.tau = tau
        self.sharpen_temp = sharpen_temp
        self.softmax = nn.Softmax(dim=1)

    def _sharpen(self, p):
        sharp_p = p**(1. / self.sharpen_temp)
        sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
        return sharp_p

    def _cross_entropy(self, x, y):
        return torch.sum(-y * torch.log(x + EPS), dim=1).mean()

    def _get_distribution(self, x, target, sharp=False):
        probabilities = self.softmax(x @ target.T / self.tau)
        if sharp:
            probabilities = self._sharpen(probabilities)
        return probabilities

    def forward(self, p1: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor,
                z2: torch.Tensor) -> torch.Tensor:

        p1 = F.normalize(p1, dim=1)
        p2 = F.normalize(p2, dim=1)
        z1 = F.normalize(z1, dim=1).detach()
        z2 = F.normalize(z2, dim=1).detach()

        supports = AllGather.apply(torch.cat([z1, z2]))

        student1 = self._get_distribution(p1, supports, sharp=False)
        teacher1 = self._get_distribution(z2, supports, sharp=True)
        student2 = self._get_distribution(p2, supports, sharp=False)
        teacher2 = self._get_distribution(z1, supports, sharp=True)

        loss1 = self._cross_entropy(student1, teacher1)
        loss2 = self._cross_entropy(student2, teacher2)

        return loss1 + loss2


def ts_simsiam_loss() -> nn.Module:
    return TSSimSiamLoss()
