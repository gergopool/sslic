import torch
import torch.nn.functional as F

from .general import Loss

__all__ = ['byol_loss']


class BYOLLoss(Loss):
    """BYOLLoss Loss"""

    def forward(self, z_t: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:
        sim = F.cosine_similarity(z_t, z_s, dim=1).mean()
        return 2 - 2 * sim


def byol_loss() -> Loss:
    return BYOLLoss