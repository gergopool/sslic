from .byol import BYOLTransform

__all__ = ['barlow_twins_transform']


class BarlowTwinsTransform(BYOLTransform):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def barlow_twins_transform(*args, **kwargs):
    return BarlowTwinsTransform(*args, **kwargs)