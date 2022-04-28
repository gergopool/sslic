from ..losses.nnclr import NNCLRLoss
from .simsiam import SimSiam
from .base_model import BaseModel

__all__ = ['nnclr_model']


class NNCLR(SimSiam):

    default_loss = NNCLRLoss

    @classmethod
    def imagenet(cls, *args, **kwargs) -> SimSiam:
        return super().imagenet(*args, dim=256, pred_dim=4096, **kwargs)

    @classmethod
    def tiny_imagenet(cls, *args, **kwargs) -> SimSiam:
        # Note: this was not mentioned in paper
        return super().tiny_imagenet(*args, dim=256, pred_dim=4096, **kwargs)

    @classmethod
    def cifar10(cls, *args, **kwargs) -> SimSiam:
        # Note: this was not mentioned in paper, pred_dim is decreased by us
        return super().cifar10(*args, dim=128, pred_dim=2048, **kwargs)

    @classmethod
    def cifar100(cls, *args, **kwargs) -> SimSiam:
        # Note: this was not mentioned in paper, pred_dim is decreased by us
        return super().cifar100(*args, dim=128, pred_dim=2048, **kwargs)


def nnclr_model() -> NNCLR:
    return NNCLR