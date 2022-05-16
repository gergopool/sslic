from .datasets import *
from torch.utils.data import Dataset
from torch import nn


def get_dataset(root: str, dataset_name: str, trans: nn.Module, is_train: bool) -> Dataset:
    """get_dataset

    Parameters
    ----------
    root : str
        Path to your dataset.
    dataset_name : str
        Name of your dataset. E.g. tiny_imagenet, cifar10, etc.
    trans : nn.Module
        The transformation that should be applied on the dataset
    split : str
        Data split you want to use. Choose from ssl, train, test.
        Note that the option 'test' will still point at the validation
        datasets, only the augmentation will be set to test mode.

    Returns
    -------
    Dataset
        Torch dataset.

    Raises
    ------
    NameError
        If either parameter is invalid and not defined. 
    """
    dataset_fn = dataset_name + "_dataset"
    if dataset_fn not in globals():
        raise NameError(f"Dataset {dataset_name} is not known.")
    return globals()[dataset_fn](root, trans, is_train)


def available_datasets():
    return set([k[:-len('_dataset')] for k in globals() if k.endswith('_dataset')])
