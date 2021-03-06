from torch import nn
from .barlow_twins import *
from .simclr import *
from .simsiam import *
from .twist import *
from .ressl import *
from .byol import *
from .mocov2 import *
from .vicreg import *
from .nnclr import *
from .experimental import *
from .lin_eval import *


def get_loss(method_name: str) -> nn.Module:
    """Initializes the loss function to the given method.

    Parameters
    ----------
    method_name : str
        Name of the method. E.g. simsiam

    Returns
    -------
    nn.Module
        The loss function as an nn.Module callable class.
    """
    method_name += "_loss"
    if method_name not in globals():
        raise NameError(f"Self-supervised loss {method_name} is unknown.")
    return globals()[method_name]()


def available_losses():
    return set([k[:-len('_loss')] for k in globals() if k.endswith('_loss')])