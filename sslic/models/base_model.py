import torch.nn as nn
from torchvision.models import resnet50
from .cifar_resnet import resnet18
from ..losses.general import Loss


class BaseModel(nn.Module):
    """General SSL base model, and abstract class which
       is useless in itself but all other SSL model inherits
       from this class.

        Parameters
        ----------
        base_encoder : nn.Module
            The backend encoder. E.g. torchvision.models.resnet50
        dim : int, optional
            The dimension of output representation, by default 128
        ssl_loss : nn.Module, optional
            The loss defined on the output representations, by default None
        n_classes : int, optional
            Number of output classes. Note that this is always needed
            because of online accuracy approximation. By default 1000
        """

    default_loss = Loss

    def __init__(self,
                 base_encoder: nn.Module,
                 dim: int = 128,
                 ssl_loss: nn.Module = None,
                 n_classes: int = 1000,
                 **kwargs):
        super(BaseModel, self).__init__()
        self.dim = dim
        self.ssl_loss = ssl_loss(emb_dim=self.dim)
        self.n_classes = n_classes

        # create the encoder
        self.encoder = base_encoder(**kwargs)
        self.prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Identity()

    @classmethod
    def imagenet(cls, *args, **kwargs):
        kwargs.setdefault("base_encoder", resnet50)
        kwargs.setdefault("ssl_loss", getattr(cls.default_loss, "imagenet"))
        kwargs.setdefault("zero_init_residual", True)
        kwargs.setdefault("n_classes", 1000)
        return cls(*args, **kwargs)

    @classmethod
    def tiny_imagenet(cls, *args, **kwargs):
        kwargs.setdefault("base_encoder", resnet18)
        kwargs.setdefault("ssl_loss", getattr(cls.default_loss, "tiny_imagenet"))
        kwargs.setdefault("n_classes", 200)
        return cls(*args, **kwargs)

    @classmethod
    def cifar10(cls, *args, **kwargs):
        kwargs.setdefault("base_encoder", resnet18)
        kwargs.setdefault("ssl_loss", getattr(cls.default_loss, "cifar10"))
        kwargs.setdefault("n_classes", 10)
        return cls(*args, **kwargs)

    @classmethod
    def cifar100(cls, *args, **kwargs):
        kwargs.setdefault("base_encoder", resnet18)
        kwargs.setdefault("ssl_loss", getattr(cls.default_loss, "cifar100"))
        kwargs.setdefault("n_classes", 100)
        return cls(*args, **kwargs)

    def step(self, progress: float):
        # Some models might require a continuous change
        assert progress >= 0. and progress <= 1.
