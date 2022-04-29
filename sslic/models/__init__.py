from torch import nn

from .barlow_twins import *
from .simclr import *
from .simsiam import *
from .ressl import *
from .byol import *
from .mocov2 import *
from .twist import *
from .vicreg import *
from .nnclr import *


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
    method_name += "_model"
    if method_name not in globals():
        raise NameError(f"Self-supervised method {method_name} is unknown.")
    model_builder = getattr(globals()[method_name](), dataset)
    return model_builder(**kwargs)