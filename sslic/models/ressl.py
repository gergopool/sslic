from ..losses.ressl import ReSSLLoss
from .momentum_model import MomentumModel

__all__ = ['ressl_model']


class ReSSL(MomentumModel):

    default_loss = ReSSLLoss

    @classmethod
    def imagenet(cls, *args, **kwargs) -> MomentumModel:
        return super().imagenet(*args, dim=512, hidden_dim=4096, momentum=0.999, **kwargs)

    @classmethod
    def tiny_imagenet(cls, *args, **kwargs) -> MomentumModel:
        return super().tiny_imagenet(*args, dim=128, hidden_dim=128, momentum=0.996, **kwargs)

    @classmethod
    def cifar10(cls, *args, **kwargs) -> MomentumModel:
        return super().cifar10(*args, dim=128, hidden_dim=128, momentum=0.99, **kwargs)

    @classmethod
    def cifar100(cls, *args, **kwargs) -> MomentumModel:
        return super().cifar100(*args, dim=128, hidden_dim=128, momentum=0.99, **kwargs)


def ressl_model() -> ReSSL:
    return ReSSL