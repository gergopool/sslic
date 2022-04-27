from .byol import BYOLTransform

__all__ = ['vicreg_transform']


def vicreg_transform(*args, **kwargs):
    '''Note
    The paper claims they used a symmetric version of BYOL's aumgnetations.
    However, in their code (https://github.com/facebookresearch/vicreg/issues/3)
    they claim that BYOL augmentations work slightly better. For simplicity,
    I also kept byol.
    '''
    return BYOLTransform(*args, **kwargs)