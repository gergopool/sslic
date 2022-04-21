from .datasets import *


def get_dataset_provider(root, dataset_name, method_name=None):
    func_name = method_name + "_datasets"
    if func_name not in globals():
        raise NameError(f"Self-supervised dataset {method_name} is unknown.")
    return globals()[func_name](root, dataset_name)
