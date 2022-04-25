from .mocov2 import MocoTransform

__all__ = ['twist_transform']


def twist_transform():
    '''Note
    This is not exactly specified in paper, but found it at their implementation
    https://github.com/bytedance/TWIST/blob/main/augmentation.py

    It is unclear if they used barlow twins augmentation, it's in their code.
    But when deciding what's considered as baseline, moco is a better choice.
    '''
    return MocoTransform(blur_chance=0.5)