from torch import nn

from .barlow_twins import *
from .simclr import *
from .simsiam import *
from .ressl import *
from .byol import *
from .mocov2 import *
from .twist import *

n_classes = {"imagenet": 1000, "cifar10": 10, "cifar100": 100}


def get_ssl_network(method_name: str, dataset: str, **kwargs) -> nn.Module:
    """Get SSL network based on name and dataset.

    Parameters
    ----------
    method_name : str
        Name of the algorithm. E.g. simsiam
    dataset : str
        Name of the dataset. E.g. cifar10

    Returns
    -------
    nn.Module
        The neural network's PyTorch module.
    """
    method_name += "_" + dataset
    if method_name not in globals():
        raise NameError(f"Self-supervised method {method_name} is unknown.")
    return globals()[method_name](**kwargs)