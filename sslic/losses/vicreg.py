import torch
import torch.nn.functional as F
from torch import nn

from .general import Loss
from ..utils import AllGather

__all__ = ['vicreg_loss']

EPS = 1e-4


class VICRegLoss(Loss):
    """VICRegLoss Loss

    Credits: https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
    """

    def __init__(self, mse_scale: float = 25., std_scale: float = 25., cov_scale: float = 1.):
        super().__init__()
        self.mse_scale = mse_scale
        self.std_scale = std_scale
        self.cov_scale = cov_scale

    def _std_loss(self, x):
        return F.relu(1 - torch.std(x)).mean()

    def _cov_loss(self, x):
        bs, dim = x.shape
        cov_m = (x.T @ x) / (bs - 1)
        return cov_m.fill_diagonal_(0).pow_(2).sum() / dim

    def gather_and_norm(self, x):
        x = torch.cat(AllGather.apply(x))
        return x - x.mean(dim=0)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:

        mse_loss = F.mse_loss(z1, z2) * self.mse_scale

        z1 = self.gather_and_norm(z1)
        z2 = self.gather_and_norm(z2)

        std_loss = (self._std_loss(z1) + self._std_loss(z2)) * 0.5 * self.std_scale
        cov_loss = self._cov_loss(z1) + self._cov_loss(z2) * self.cov_scale

        return mse_loss + std_loss + cov_loss


def vicreg_loss() -> Loss:
    return VICRegLoss
