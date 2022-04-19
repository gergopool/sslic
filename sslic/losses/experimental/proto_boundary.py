import torch
from torch import nn
import torch.nn.functional as F
import random

from ...utils import AllGather, after_init_world_size_n_rank

__all__ = [
    'proto_boundary_loss',
    'proto_boundary_cifar10_loss',
    'cluster_boundary_loss',
    'cluster_boundary_cifar10_loss'
]

EPS = 1e-6


class ProtoBoundaryLoss(nn.Module):

    def __init__(self,
                 emb_dim: int = 512,
                 n_clusters: int = 8192,
                 cluster_size: int = 8,
                 tau_s: float = 0.1,
                 tau_t: float = 0.05):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_clusters = n_clusters
        self.cluster_size = cluster_size
        self.tau_s = tau_s
        self.tau_t = tau_t

        self.world_size, self.rank = after_init_world_size_n_rank()

        self.protos = nn.Parameter(torch.rand(self.n_clusters, self.emb_dim) - 0.5)
        self.register_buffer("queue", torch.randn(self.n_clusters, self.cluster_size, self.emb_dim))
        self.register_buffer("queue_idx", torch.zeros(self.n_clusters, dtype=torch.int64))
        self.register_buffer("visit_count", torch.zeros(self.n_clusters, dtype=torch.int64))
        self.criterion = nn.CrossEntropyLoss()

    @property
    def memory_filled(self):
        return (self.visits >= self.n_clusters).all()

    @torch.no_grad()
    def add_to_queue(self, z: torch.Tensor, y: torch.Tensor) -> None:

        batch_z = AllGather.apply(z)
        batch_y = AllGather.apply(y)

        self.visit_count -= self.visit_count.min()
        sample_chances = F.softmax(self.visit_count.float(), dim=-1)
        sample_chances -= 1 / (self.n_clusters + 1)
        should_sample = torch.tensor([random.random() < sample_chances[y] for y in batch_y
                                     ]).to(z.device)

        if self.world_size > 1:
            torch.distributed.broadcast(should_sample, src=0)

        for z, y, is_sampled in zip(batch_z, batch_y, should_sample):
            if is_sampled:
                y = self.visit_count.argmin()
            self.queue[y, self.queue_idx[y]] = z
            self.queue_idx[y] = (self.queue_idx[y] + 1) % self.cluster_size
            self.visit_count[y] += 1

    def cross_entropy(self, x, y):
        return torch.sum(-y * torch.log(x + EPS), dim=1).mean()

    def forward(self, z_t: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:

        z_t = F.normalize(z_t, dim=1)
        z_s = F.normalize(z_s, dim=1)
        protos = F.normalize(self.protos)

        # Loss for encoder gradients
        p_s = F.softmax(z_s @ protos.T.detach() / self.tau_s, dim=1)
        p_t = F.softmax(z_t @ protos.T.detach() / self.tau_t, dim=1).detach()
        encoder_labels = p_t.argmax(dim=1).detach()
        encoder_loss = self.cross_entropy(p_s, p_t)

        # Loss for proto gradients
        queue = self.queue.clone().detach()
        proto_labels = torch.arange(self.n_clusters).repeat_interleave(self.cluster_size)
        proto_labels = proto_labels.detach().to(queue.device)
        preds = F.normalize(queue.view(-1, self.emb_dim), dim=1) @ protos.T / self.tau_s
        proto_loss = self.criterion(preds, proto_labels)

        loss = encoder_loss + proto_loss

        self.add_to_queue(z_t, encoder_labels)

        return loss


class ClusterBoundaryLoss(ProtoBoundaryLoss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.protos = None

    def cross_entropy(self, x, y):
        return torch.sum(-y * torch.log(x + EPS), dim=1).mean()

    def forward(self, z_t: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:
        z_t = F.normalize(z_t, dim=1)
        z_s = F.normalize(z_s, dim=1)
        queue = F.normalize(self.queue, dim=2).clone().detach().view(-1, self.emb_dim)

        labels = F.softmax(z_t @ queue.T / self.tau / 0.5, dim=1).detach()
        labels = labels.view(-1, self.n_clusters, self.cluster_size).sum(dim=-1)

        # preds = (z_s @ queue.T / self.tau).view(-1, self.n_clusters, self.cluster_size).mean(dim=-1)
        # loss = self.criterion(preds, labels)
        # self.add_to_queue(z_t, labels)

        preds = F.softmax(z_s @ queue.T / self.tau, dim=1)
        preds = preds.view(-1, self.n_clusters, self.cluster_size).sum(dim=-1)
        # one_hot_labels = torch.eye(self.n_clusters)[labels].to(preds.device)
        loss = self.cross_entropy(preds, labels)

        self.add_to_queue(z_t, labels.argmax(dim=-1))

        return loss


def proto_boundary_loss(*args, **kwargs):
    return ProtoBoundaryLoss(*args, **kwargs)


def proto_boundary_cifar10_loss(*args, **kwargs):
    return ProtoBoundaryLoss(128, *args, n_clusters=64, cluster_size=32, **kwargs)


def cluster_boundary_loss(*args, **kwargs):
    return ClusterBoundaryLoss(*args, **kwargs)


def cluster_boundary_cifar10_loss(*args, **kwargs):
    return ClusterBoundaryLoss(128, *args, n_clusters=64, cluster_size=32, **kwargs)