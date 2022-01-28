import torch

def info_nce_loss(z1, z2, tau=0.1):

    # Batch size
    n = len(z1)

    # Labels telling which images make pairs
    e = torch.eye(n).to(z1.device)
    labels = torch.cat((
        torch.cat((e, e)).T,
        torch.cat((e, e), dim=1)
    ))

    # Combine views and normalize
    z = torch.cat((z1, z2), dim=0)
    z = torch.nn.functional.normalize(z, dim=1)

    # Note: The following code might require a large amount of memory
    # in case of large batch size
    sim_m = z @ z.T / tau

    # Removing diagonal
    mask = torch.eye(2 * n, dtype=torch.bool).to(z.device)
    labels = labels[~mask].view(2 * n, 2 * n-1)
    sim_m = sim_m[~mask].view(2 * n, 2 * n-1)

    # Get probability distribution
    sim_m = torch.nn.functional.softmax(sim_m, dim=1)

    # Choose values on which we calculate the loss
    positive_p = torch.masked_select(sim_m, labels.bool())
    loss = -torch.sum(torch.log(positive_p)) / n

    return loss

def barlow_twins_loss(z1, z2, lambd=5e-3):

    batch_size = len(z1)

    c = z1.T @ z2 / batch_size
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    loss = on_diag + lambd * off_diag

    # Note: Since the loss grows with the number of dimensions we choose,
    # I've decided to scale it down a bit. This is different from facebookresearch's code.
    loss /= z1.shape[1] / 10
    return loss

def simsiam_loss(p1, p2, z1, z2):
    loss = (cos_sim(p1, z2) + cos_sim(p2, z1)) / 2.
    return loss

def cos_sim(x1, x2):
    x1 = torch.nn.functional.normalize(x1, dim=1)
    x2 = torch.nn.functional.normalize(x2, dim=1)
    return torch.sum(x1 * x2) / len(x1)

def off_diagonal(x):
    n = len(x)
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
