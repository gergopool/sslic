import torch
import torch.nn as nn


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

    def __init__(self,
                 base_encoder: nn.Module,
                 dim: int = 128,
                 ssl_loss: nn.Module = None,
                 n_classes: int = 1000,
                 **kwargs):
        super(BaseModel, self).__init__()
        self.dim = dim
        self.ssl_loss = ssl_loss
        self.n_classes = n_classes

        # create the encoder
        self.encoder = base_encoder(**kwargs)
        self.prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Identity()

