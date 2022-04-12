import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['proto_loss']


class ProtoLoss(nn.Module):

    def __init__(self, emb_dim=2048, n_classes=2048, n_units=1, tau=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_classes = n_classes
        self.n_units = n_units
        self.tau = tau

        w = torch.rand(n_classes * n_units, emb_dim)
        self.protos = nn.Parameter(w, requires_grad=True)

        self.labels = torch.eye(n_classes).repeat_interleave(n_units, dim=0)
        self.softmax = torch.nn.Softmax(dim=1)

    def cross_entropy(self, x, y):
        x = self.softmax(x) @ self.labels
        y = self.softmax(y) @ self.labels
        return torch.mean(-y * torch.log(x))

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, p1: torch.Tensor,
                p2: torch.Tensor) -> torch.Tensor:

        protos = F.normalize(self.protos)

        sim1 = p1 @ protos.T / self.tau
        sim2 = p2 @ protos.T / self.tau
        sim1_target = z2 @ protos.T / self.tau / 0.5
        sim2_target = z1 @ protos.T / self.tau / 0.5

        # sim1 = self.protos(p1)
        # sim2 = self.protos(p2)
        # sim1_target = self.protos(z2) / 0.5
        # sim2_target = self.protos(z1) / 0.5

        loss = (self.cross_entropy(sim1, sim1_target.detach()) +
                self.cross_entropy(sim2, sim2_target.detach())) * 0.5

        return loss


def proto_loss(*args, **kwargs):
    return ProtoLoss(*args, **kwargs)