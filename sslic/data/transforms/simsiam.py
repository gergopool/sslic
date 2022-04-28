from .mocov2 import MocoTransform

__all__ = ['simsiam_transform']


def simsiam_transform():
    return MocoTransform(blur_chance=0.5)