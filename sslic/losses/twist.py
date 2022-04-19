import torch
from torch import nn
import torch.nn.functional as F
from ..utils import AllGather

__all__ = ['twist_loss']

EPS = 1e-5


class TwistLoss(nn.Module):

    def __init__(self, lam1: float = 1., lam2: float = 1.):
        super(TwistLoss, self).__init__()
        self.lam1 = lam1
        self.lam2 = lam2

    def forward(self, p1: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor,
                z2: torch.Tensor) -> torch.Tensor:
        loss1 = self.loss(p1, z2)
        loss2 = self.loss(p2, z1)
        return (loss1 + loss2) * 0.5

    def loss(self, p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        probs1 = F.softmax(p, dim=-1)
        probs2 = F.softmax(z.detach(), dim=-1)

        kl_term = self.KL(probs1, probs2) + self.KL(probs2, probs1)
        eh_term = self.EH(probs1) + self.EH(probs2) * self.lam1
        he_term = self.HE(probs1) + self.HE(probs2) * self.lam2

        loss = (kl_term + eh_term - he_term) * 0.5
        return loss

    def KL(self, probs1: torch.Tensor, probs2: torch.Tensor) -> torch.Tensor:
        return (probs1 * (probs1 + EPS).log() - probs1 * (probs2 + EPS).log()).sum(dim=1).mean()

    def HE(self, probs: torch.Tensor) -> torch.Tensor:
        global_probs = AllGather.apply(probs)
        world_size = len(global_probs) / len(probs)
        mean = global_probs.mean(dim=0)
        entropy = -(mean * (mean + world_size * EPS).log()).sum()
        return entropy

    def EH(self, probs: torch.Tensor) -> torch.Tensor:
        return -(probs * (probs + EPS).log()).sum(dim=1).mean()


def twist_loss() -> nn.Module:
    return TwistLoss()