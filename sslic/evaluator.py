from tkinter.ttk import LabeledScale
import torch
from torch import nn
import torch.nn.functional as F


class SnnEvaluator(nn.Module):
    def __init__(self, dim, n_classes, max_queue_size=10):
        super(SnnEvaluator, self).__init__()
        self.dim = dim
        self.n_classes = n_classes
        self.max_queue_size = max_queue_size
        self.memory_bank = nn.Parameter(torch.zeros((n_classes, max_queue_size, dim)))

        labels = torch.arange(n_classes).repeat_interleave(max_queue_size)
        self.onehot_labels = nn.Parameter(F.one_hot(labels, n_classes).float())    

    def update(self, batch_x, batch_y):
        batch_x = batch_x.detach()
        for (x, y) in zip(batch_x, batch_y):
            new_values = torch.cat((self.memory_bank[y][1:], x.unsqueeze(0)))
            with torch.no_grad():
                self.memory_bank[y] *= 0
                self.memory_bank[y] += new_values

    def forward(self, x, y):

        # Similarity matrix
        train_x = F.normalize(self.memory_bank, dim=1).view(-1, self.dim)
        val_x = F.normalize(x, dim=1)
        sim_m = F.softmax(val_x @ train_x.T, dim=1) @ self.onehot_labels

        # Accuracy
        top5_preds = sim_m.topk(5, dim=1)[1]
        top1_preds = top5_preds[:, 0]
        top5_labels = y.repeat_interleave(5).view(-1, 5)
        top5_acc = (top5_preds == top5_labels).any(dim=1).sum() / len(y)
        top1_acc = (top1_preds == y).sum() / len(y)
        
        return top1_acc, top5_acc