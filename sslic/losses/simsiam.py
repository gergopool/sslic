import torch
import torch.nn.functional as F
from torch import nn

__all__ = ['simsiam_loss']


class SimSiamLoss(nn.Module):
    """SimSiam Loss"""

    def forward(self, p1: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor,
                z2: torch.Tensor) -> torch.Tensor:
        sim1 = F.cosine_similarity(p1, z2.detach(), dim=1).mean()
        sim2 = F.cosine_similarity(p2, z1.detach(), dim=1).mean()
        loss = -(sim1 + sim2) / 2.
        return loss


def simsiam_loss() -> nn.Module:
    return SimSiamLoss()
