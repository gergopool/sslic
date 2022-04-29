from .byol import BYOLTransform

__all__ = ['nnclr_transform']


class NNCLRTransform(BYOLTransform):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def nnclr_transform(*args, **kwargs):
    return NNCLRTransform(*args, **kwargs)