import torch
import torch.nn.functional as F
from typing import List

from .general import Loss

__all__ = ['simsiam_loss']


class SimSiamLoss(Loss):
    """SimSiam Loss"""

    def forward(self,
                p1: torch.Tensor,
                p2: torch.Tensor,
                z1: torch.Tensor,
                z2: torch.Tensor,
                crops: List[torch.Tensor]) -> torch.Tensor:
        sim1 = F.cosine_similarity(p1, z2.detach(), dim=1).mean()
        sim2 = F.cosine_similarity(p2, z1.detach(), dim=1).mean()
        loss = -(sim1 + sim2) / 2.

        if len(crops):
            # NOTE
            # This is not covered in simsiam paper. This is a multicrop
            # option in order to compare with other methods
            n_views = len(crops) // len(z1)
            z = (z1 + z2).repeat(n_views, 1)
            loss = (loss * 2 + F.cosine_similarity(crops, z.detach(), dim=1).mean()) / 3.

        return loss


def simsiam_loss() -> Loss:
    return SimSiamLoss
