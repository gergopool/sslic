import torch.nn as nn

from ..losses.mocov2 import Mocov2Loss

from .momentum_model import MomentumModel

__all__ = ['mocov2_model']
'''
Important note

Momentum parameteres were not reported for datasets other than Imagenet. Therefore,
for convinient comparisons, we maintain the same momentum parameters that ReSSL
uses for smaller datasets.
'''


class MocoV2(MomentumModel):

    default_loss = Mocov2Loss

    @classmethod
    def imagenet(cls, *args, **kwargs) -> MomentumModel:
        return super().imagenet(*args, dim=128, hidden_dim=128, momentum=0.999, **kwargs)

    @classmethod
    def tiny_imagenet(cls, *args, **kwargs) -> MomentumModel:
        # Note: This is undeclared in paper and therefore using metrics suggested in ressl
        return super().tiny_imagenet(*args, dim=128, hidden_dim=128, momentum=0.996, **kwargs)

    @classmethod
    def cifar10(cls, *args, **kwargs) -> MomentumModel:
        # Note: This is undeclared in paper and therefore using metrics suggested in ressl
        return super().tiny_imagenet(*args, dim=128, hidden_dim=128, momentum=0.99, **kwargs)

    @classmethod
    def cifar100(cls, *args, **kwargs) -> MomentumModel:
        # Note: This is undeclared in paper and therefore using metrics suggested in ressl
        return super().tiny_imagenet(*args, dim=128, hidden_dim=128, momentum=0.99, **kwargs)


def mocov2_model() -> MocoV2:
    return MocoV2