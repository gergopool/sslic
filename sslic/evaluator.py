import torch
from torch import nn
import torch.nn.functional as F


class SnnEvaluator(nn.Module):
    """Soft Nearest Neighbor Evaluator

    This class saves the last N representations of a class in a FIFO
    queue. When evaluating a new batch of representations, it calculates
    the cosine similarities to each representation saved in the FIFO.
    Then, sums up the similarities per each class and predicts the class
    which achieved the highest sum of cosine similarities.

    Parameters
        ----------
        dim : int
            Dimension of representations.
        n_classes : int
            Number of classes.
        max_queue_size : int, optional
            Number of representation per class in queue, by default 10
    """

    def __init__(self, dim: int, n_classes: int, max_queue_size: int = 10):
        super(SnnEvaluator, self).__init__()
        self.dim = dim
        self.n_classes = n_classes
        self.max_queue_size = max_queue_size

        # This will store the last N representations seen per class
        self.memory_bank = nn.Parameter(torch.zeros((n_classes, max_queue_size, dim)))

        # Labels of the memory bank
        labels = torch.arange(n_classes).repeat_interleave(max_queue_size)
        self.onehot_labels = nn.Parameter(F.one_hot(labels, n_classes).float())

    def update(self, batch_x: torch.Tensor, batch_y: torch.Tensor):
        """Save new embeddings in memory along with their classes.

        Parameters
        ----------
        batch_x : torch.Tensor
            The embeddings of images.
        batch_y : torch.Tensor
            The labels of the embeddings. Note that all y values must
            be less than the number of classes given at initialisation.
        """
        batch_x = batch_x.detach()
        for (x, y) in zip(batch_x, batch_y):
            # FIFO push
            new_values = torch.cat((self.memory_bank[y][1:], x.unsqueeze(0)))
            with torch.no_grad():
                self.memory_bank[y] *= 0
                self.memory_bank[y] += new_values

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate top1 accuracy, given the representations and
           their corresponding labels.

        Parameters
        ----------
        x : torch.Tensor
            The representations, embeddings of images.
        y : torch.Tensor
            The target labels.

        Returns
        -------
        torch.Tensor
            Accuracy based on soft-neares-neighbor method.
        """

        # Similarity matrix
        train_x = F.normalize(self.memory_bank.view(-1, self.dim), dim=1)
        val_x = F.normalize(x, dim=1)
        sim_m = F.softmax(val_x @ train_x.T, dim=1) @ self.onehot_labels

        # Accuracy
        top5_preds = sim_m.topk(5, dim=1)[1]
        top1_preds = top5_preds[:, 0]
        top5_labels = y.repeat_interleave(5).view(-1, 5)
        top5_acc = (top5_preds == top5_labels).any(dim=1).sum() / len(y)
        top1_acc = (top1_preds == y).sum() / len(y)

        # Note: top5 will be added later
        return top1_acc