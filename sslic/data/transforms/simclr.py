from .mocov2 import MocoTransform

__all__ = ['simclr_transform']


def simclr_transform():
    return MocoTransform(blur_chance=0.5)