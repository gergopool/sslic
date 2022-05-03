import torch.nn as nn
from torch.nn.modules.module import _addindent

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
                 sync_batchnorm: bool = True,
                 **kwargs):
        super(BaseModel, self).__init__()
        self.dim = dim
        self.ssl_loss = ssl_loss(emb_dim=self.dim)
        self.n_classes = n_classes
        self.sync_batchnorm = sync_batchnorm

        # create the encoder
        self.encoder = base_encoder(**kwargs)
        self.prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Identity()

    @classmethod
    def imagenet(cls, *args, **kwargs):
        kwargs.setdefault("base_encoder", resnet50)
        kwargs.setdefault("ssl_loss", cls.default_loss)
        kwargs.setdefault("zero_init_residual", True)
        kwargs.setdefault("n_classes", 1000)
        kwargs['ssl_loss'] = kwargs['ssl_loss'].imagenet
        return cls(*args, **kwargs)

    @classmethod
    def tiny_imagenet(cls, *args, **kwargs):
        kwargs.setdefault("base_encoder", resnet18)
        kwargs.setdefault("ssl_loss", cls.default_loss)
        kwargs.setdefault("n_classes", 200)
        kwargs.setdefault("pool", True)  # Apply pool on resnet18
        kwargs['ssl_loss'] = kwargs['ssl_loss'].tiny_imagenet
        return cls(*args, **kwargs)

    @classmethod
    def cifar10(cls, *args, **kwargs):
        kwargs.setdefault("base_encoder", resnet18)
        kwargs.setdefault("ssl_loss", cls.default_loss)
        kwargs.setdefault("n_classes", 10)
        kwargs['ssl_loss'] = kwargs['ssl_loss'].cifar10
        return cls(*args, **kwargs)

    @classmethod
    def cifar100(cls, *args, **kwargs):
        kwargs.setdefault("base_encoder", resnet18)
        kwargs.setdefault("ssl_loss", cls.default_loss)
        kwargs.setdefault("n_classes", 100)
        kwargs['ssl_loss'] = kwargs['ssl_loss'].cifar100
        return cls(*args, **kwargs)

    def step(self, progress: float):
        # Some models might require a continuous change
        assert progress >= 0. and progress <= 1.

    # =====================================================================
    # SUQEEZE THE __REPR__ OF BACKEND
    # =====================================================================

    def __repr__(self):
        '''__repr__

        This function is a copy of the original pytorch __repr__ implementation,
        except it skips the __repr__ of the backend/encoder, so we only see 
        the relevant modules in the standard output.
        '''

        # Current string form
        string_form = super().__repr__()

        if len(string_form) < 6:
            return string_form

        # Stack for counting parenthesis
        stack = 0

        # Output stirng
        new_string = string_form[:6]

        for i in range(6, len(string_form)):
            c = string_form[i]
            if string_form[i - 6:i].lower() == 'resnet' or stack != 0:
                # Resnet found, wait until parenthesis are closed
                stack += int(c == '(') - int(c == ')')
            else:
                # Add character
                new_string += c

        # Add which backend is currently used exactly
        backend_name = ['ResNet50()', 'ResNet18()'][self.encoder.conv1.kernel_size[0] == 3]
        new_string = new_string.replace('ResNet', backend_name)

        return new_string