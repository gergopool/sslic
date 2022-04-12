import torch
from torch import embedding_renorm_, nn
import math
import torch.nn.functional as F

from torchvision.ops import sigmoid_focal_loss

from ..utils import AllGather

__all__ = ['arc_simclr_loss']


# From https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/blob/master/src/modeling/metric_learning.py
# Added type annotations, device, and 16bit support
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau: float = 30,
        m: float = 0.5,
        easy_margin: bool = False,
        smoothing: float = 0,
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tau = tau
        self.m = m
        self.smoothing = smoothing  # label smoothing
        # self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self,
                input: torch.Tensor,
                label: torch.Tensor,
                device: str = "cuda") -> torch.Tensor:
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.normalize(input, dim=1) @ F.normalize(self.weights.detach(), dim=1).T
        # Enable 16 bit precision
        cosine = cosine.to(torch.float32)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        batch_size = input.shape[0]
        # label = torch.eye(batch_size).to(input.device)

        if self.smoothing:
            label = label * (1 - self.smoothing) + self.smoothing / batch_size

        label = (label * phi) + ((1.0 - label) * cosine)
        output = label * self.tau

        return output


class ArcSimCLR(nn.Module):

    def __init__(self, tau: float = 0.07, m: float = 0.5, smoothing: float = 0):

        super(ArcSimCLR, self).__init__()

        self.tau = tau
        self.m = torch.tensor(m)
        self.smoothing = smoothing
        self.cos_m = torch.cos(self.m)
        self.sin_m = torch.sin(self.m)
        pi = torch.tensor(math.pi)
        self.th = torch.cos(pi - self.m)
        self.mm = torch.sin(pi - self.m) * self.m

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:

        # Collect from all gpu
        z1 = AllGather.apply(z1)
        z2 = AllGather.apply(z2)

        # Combine views and normalize
        z = torch.cat((z1, z2), dim=0)
        z = F.normalize(z, dim=1)
        n = len(z)

        # Labels telling which images make pairs
        ones = torch.ones(n // 2).to(z.device)
        labels = ones.diagflat(n // 2) + ones.diagflat(-n // 2)

        cosine = z @ z.T
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi.half(), (cosine - self.mm).half())

        if self.smoothing:
            labels = labels * (1 - self.smoothing) + self.smoothing / n

        sim_m = (labels * phi) + ((1.0 - labels) * cosine)
        sim_m = sim_m.fill_diagonal_(-1) / self.tau

        # loss = sigmoid_focal_loss(sim_m, labels, reduction='mean')

        # Get probability distribution
        sim_m = torch.nn.functional.log_softmax(sim_m, dim=1)

        # Choose values on which we calculate the loss
        loss = -torch.sum(sim_m * labels) / n

        return loss


class ArcLoss(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        # self.criterion = FocalLoss()
        self.arc = ArcMarginProduct(*args, **kwargs)

    def forward(self, embeddings, targets):
        outputs = self.arc(embeddings, targets, embeddings.device)
        n = self.arc.out_features
        onehot_targets = torch.eye(n)[targets].to(embeddings.device)
        loss = sigmoid_focal_loss(outputs, onehot_targets, reduction='mean')
        return loss


def arc_simclr_loss():
    return ArcSimCLR()