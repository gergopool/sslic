import torch
from .utils import AllGather

def info_nce_loss(z1, z2, tau=0.5):

    # Normalzie
    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = torch.nn.functional.normalize(z1, dim=1)

    # Collect from all gpu
    z1 = AllGather.apply(z1)
    z2 = AllGather.apply(z2)

    # Combine views and normalize
    z = torch.cat((z1, z2), dim=0)
    n = len(z)

    # Labels telling which images make pairs
    ones = torch.ones(n//2).to(z.device)
    labels = ones.diagflat(n//2) + ones.diagflat(-n//2)

    # Note: The following code might require a large amount of memory
    # in case of large batch size
    sim_m = z @ z.T

    # This is a bit of cheat. Instead of removing cells from
    # the matrix where i==j, instead we set it to a very small value
    sim_m = sim_m.fill_diagonal_(-1) / tau

    # Get probability distribution
    sim_m = torch.nn.functional.log_softmax(sim_m, dim=1)

    # Choose values on which we calculate the loss
    loss = -torch.sum(sim_m * labels) / n

    return loss

def barlow_twins_loss(z1, z2, lambd=5e-3):

    batch_size = len(z1)

    c = z1.T @ z2 / batch_size
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    loss = on_diag + lambd * off_diag

    # Note: Since the loss grows with the number of dimensions we choose,
    # I've decided to scale it down a bit. This is different from facebookresearch's code.
    #loss /= z1.shape[1] / 10
    return loss

def simsiam_loss(p1, p2, z1, z2):
    loss = (neg_cos_sim(p1, z2) + neg_cos_sim(p2, z1)) / 2.
    return loss

def neg_cos_sim(x1, x2):
    return -torch.nn.functional.cosine_similarity(x1, x2, dim=1).mean()

def off_diagonal(x):
    n = len(x)
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
