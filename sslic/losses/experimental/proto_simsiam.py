import torch
from torch import nn
import torch.nn.functional as F

from ...utils import AllGather

__all__ = ['proto_simsiam_loss']

EPS = 1e-6


class ProtoSimSiamLoss(nn.Module):

    def __init__(self, emb_dim=2048, n_classes=1024, n_units=1, tau=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_classes = n_classes
        self.n_units = n_units
        self.tau = tau

        w = torch.rand(n_classes * n_units, emb_dim) - 0.5
        self.protos = nn.Parameter(w, requires_grad=True)
        self.labels = nn.Parameter(torch.eye(n_classes).repeat_interleave(n_units, dim=0),
                                   requires_grad=False)
        self.softmax = nn.Softmax(dim=1)

    def cross_entropy(self, x, y):
        x = self.softmax(x) @ self.labels
        y = self.softmax(y) @ self.labels
        y = y.detach()
        return torch.sum(-y * torch.log(x + EPS), dim=1).mean()

    def memax(self, x):
        x = self.softmax(x / 0.5) @ self.labels
        x = AllGather.apply(x)
        p = x.mean(dim=0)
        return torch.sum(-p * torch.log(p + EPS))

    def forward(self, p1: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor,
                z2: torch.Tensor) -> torch.Tensor:

        protos = F.normalize(self.protos)

        sim1 = p1 @ protos.T / self.tau
        sim2 = p2 @ protos.T / self.tau
        sim1_target = z2 @ protos.T / self.tau / 0.5
        sim2_target = z1 @ protos.T / self.tau / 0.5

        celoss = (self.cross_entropy(sim1, sim1_target) +
                  self.cross_entropy(sim2, sim2_target)) * 0.5
        memax = (self.memax(sim1) + self.memax(sim2)) * 0.5

        return celoss - memax


def proto_simsiam_loss(*args, **kwargs):
    return ProtoSimSiamLoss(*args, **kwargs)