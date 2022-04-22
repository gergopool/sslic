from torchvision.transforms import Normalize
from typing import Callable

NORMS = {
    "imagenet": {
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]
    },
    "tiny_imagenet": {
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]
    },
    "cifar10": {
        "mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010]
    },
    "cifar100": {
        "mean": [0.5071, 0.4867, 0.4408], "std": [0.2675, 0.2565, 0.2761]
    }
}


def normalize(dataset_name: str) -> Callable:
    '''Get normalization transformation of dataset'''
    if dataset_name not in NORMS:
        raise NameError(f"No norm values have been defined to dataset {dataset_name}")
    norm_data = NORMS[dataset_name]
    return Normalize(**norm_data)