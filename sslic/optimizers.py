import torch
from torch import nn
from sslic.lars import LARS

__all__ = ['get_optimizer']


def get_optimizer(method_name: str,
                  model: nn.Module,
                  batch_size: int = None,
                  lr: float = None,
                  **kwargs) -> torch.optim.Optimizer:
    """get_optimizer

    This function calculates the learning rates based on batch size and returns
    with the optimzier of your desired method. You can override the default lr
    calculation by providing a learning rate on your own.

    Parameters
    ----------
    method_name : str
        Name of the algorithm. E.g. simsiam
    batch_size : int
        Size of batch summed up on all gpus.
    lr : float
        Learning rate. If you provide this, the autmatic learning rate calculation is ignored.

    Returns
    -------
    torch.optim.Optimizer
        The otpimizer
    """
    if not batch_size and not lr:
        raise ValueError("You must provide either the batch size or your custom learning rate.")

    if not lr:
        lr = _base_lr(method_name, batch_size)

    method_name = "_" + method_name
    if method_name not in globals():
        raise NameError(f"Self-supervised method {method_name} is unknown.")
    return globals()[method_name](model, lr=lr)


def _base_lr(mode: str, batch_size: int):
    scale = batch_size / 256
    lrs = {
        "simsiam": scale * 0.1,
        "simclr": scale * 0.3,
        "barlow_twins": scale,
        "ressl": scale * 0.05,
        "byol": scale * 0.2,
        "mocov2": scale * 0.03,
        "twist": scale * 0.5,
        "vicreg": scale * 0.2,
    }
    return lrs[mode]


def _vicreg(*args, **kwargs):
    return _byol(*args, **kwargs)


def _twist(*args, **kwargs):
    return _byol(*args, **kwargs)


def _barlow_twins(*args, **kwargs):
    return _byol(*args, **kwargs)


def _byol(model: nn.Module, lr: float, weight_decay: float = 1.5e-6):
    optim_params = []
    for name, param in model.named_parameters():
        if ('bn' in name or 'downsample.1' in name or 'bias' in name):
            param_dict = {'params': param, 'weight_decay': 0., 'lars_exclude': True}
        else:
            param_dict = {'params': param}
        optim_params.append(param_dict)
    sgd = torch.optim.SGD(optim_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    return LARS(sgd)


def _ressl(*args, **kwargs):
    return _mocov2(*args, weight_decay=5e-4, **kwargs)


def _simclr(model: nn.Module, lr: float, weight_decay: float = 1e-6):
    optim_params = [{"params": model.parameters(), 'fix_lr': False}]
    sgd = torch.optim.SGD(optim_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    return LARS(sgd)


def _mocov2(model: nn.Module, lr: float, weight_decay: float = 1e-4):
    optim_params = [{"params": model.parameters()}]
    return torch.optim.SGD(optim_params, lr=lr, momentum=0.9, weight_decay=weight_decay)


def _simsiam(model: nn.Module, lr: float, weight_decay: float = 1e-4):
    optim_params = []
    for name, param in model.named_parameters():
        if 'predictor' in name:
            optim_params.append({'params': param, 'fix_lr': True})
        else:
            optim_params.append({'params': param, 'fix_lr': False})

    return torch.optim.SGD(optim_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
