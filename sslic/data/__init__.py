from .datasets import *
from torch.utils.data import Dataset


def get_dataset(root: str, method_name: str, dataset_name: str, split: str, **kwargs) -> Dataset:
    """get_dataset

    Parameters
    ----------
    root : str
        Path to your dataset.
    method_name : str
        Name of your ssl method. E.g. byol, simsiam, etc.
    dataset_name : str
        Name of your dataset. E.g. tiny_imagenet, cifar10, etc.
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
    return globals()[dataset_fn](root, method_name, split, **kwargs)