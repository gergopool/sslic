from .mocov2 import *
from .barlow_twins import *
from .ressl import *
from .simsiam import *
from .byol import *
from .mocov2 import *
from .twist import *
from .vicreg import *


def get_transform(method_name: str, dataset_name: str, split: str):

    # Transform generator class
    transform_gen = method_name + '_transform'
    if transform_gen not in globals():
        raise NameError(f"Transformation for {method_name} is not unknown.")

    return globals()[transform_gen]().make_by_dataset(dataset_name, split)
