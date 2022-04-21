from .general import GeneralTransform
from .moco import MocoTransform
from .barlow_twins import BarlowTwinsTransform
from .ressl import ReSSLTransform


def get_transform_generator(method_name: str) -> GeneralTransform:

    if method_name in ['simsiam', 'simclr']:
        return MocoTransform()
    elif method_name == 'barlow_twins':
        return BarlowTwinsTransform()
    elif method_name == 'ressl':
        return ReSSLTransform()
    else:
        raise NameError(f"Unknown method name: {method_name}")