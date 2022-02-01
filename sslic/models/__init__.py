from .barlow_twins import *
from .simclr import *
from .simsiam import *

n_classes = {"imagenet": 1000, "cifar10": 10, "cifar100": 100}

def get_ssl_method(method_name, dataset, **kwargs):
    method_name += "_"+dataset
    if method_name not in globals():
        raise NameError(f"Self-supervised method {method_name} is unknown.")
    return globals()(**kwargs)