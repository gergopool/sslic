import torch
from torch import nn
import torch.nn.functional as F

from ...utils import AllGather

__all__ = ['proto_boundary_loss', 'proto_boundary_cifar10_loss']

EPS = 1e-6


class ProtoBoundaryLoss(nn.Module):

    def __init__(self, emb_dim: int = 512, n_clusters: int = 2048, cluster_size: int = 8):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_clusters = n_clusters
        self.cluster_size = cluster_size

        self.protos = nn.Linear(emb_dim, n_clusters)
        self.register_buffer("queue", torch.randn(self.n_clusters, self.cluster_size, self.emb_dim))
        self.register_buffer("queue_idx", torch.zeros(self.n_clusters, dtype=torch.int64))
        self.register_buffer("visit_count", torch.zeros(self.n_clusters, dtype=torch.int64))
        self.criterion = nn.CrossEntropyLoss()

    # def cross_entropy(self, x, y):
    #     return torch.sum(-y * torch.log(x + EPS), dim=1).mean()

    @property
    def memory_filled(self):
        return (self.visits >= self.n_clusters).all()

    @torch.no_grad()
    def add_to_queue(self, z, y):
        z = AllGather.apply(z)
        y = AllGather.apply(y)

        over_pred_y = torch.where(torch.bincount(y) > self.cluster_size, True, False).ravel()

        for i, mask in enumerate(over_pred_y):
            if not mask:
                self.visit_count[i] += 1

        count = 0
        while over_pred_y.any():
            for i, mask in enumerate(over_pred_y):
                if mask:
                    continue
                new_label = self.visit_count.argmin()
                y[i] = new_label
                self.visit_count[new_label] += 1
            # print(torch.bincount(y))
            over_pred_y = torch.where(torch.bincount(y) > self.cluster_size, True, False).ravel()

            count += 1
            if count >= 200:
                raise KeyboardInterrupt("Infinite loop")

        for item, cluster_i in zip(z, y):
            item_i = self.queue_idx[cluster_i]
            self.queue[cluster_i, item_i] = item.detach()
            self.queue_idx[cluster_i] = (self.queue_idx[cluster_i] + 1) % self.cluster_size
            # self.visit_count[cluster_i] += 1

    def forward(self, z_t: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:

        # Loss for encoder gradients
        with torch.no_grad():
            p_s = self.protos(z_s)
            p_t = self.protos(z_t)
            encoder_labels = p_t.argmax(dim=1).detach()
        encoder_loss = self.criterion(p_s, encoder_labels)

        # Loss for proto gradients
        queue = self.queue.clone().detach()
        proto_labels = torch.arange(self.n_clusters).repeat_interleave(self.cluster_size)
        proto_labels = proto_labels.detach().to(queue.device)
        preds = self.protos(queue.view(-1, self.emb_dim))
        proto_loss = self.criterion(preds, proto_labels)

        loss = encoder_loss + proto_loss

        self.add_to_queue(z_t, encoder_labels)

        return loss


def proto_boundary_loss(*args, **kwargs):
    return ProtoBoundaryLoss(*args, **kwargs)


def proto_boundary_cifar10_loss(*args, **kwargs):
    return ProtoBoundaryLoss(128, *args, n_clusters=64, cluster_size=8, **kwargs)