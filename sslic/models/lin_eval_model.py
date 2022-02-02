import torch.nn as nn

from ..losses import barlow_twins_loss
from .base_model import BaseModel
from torchvision import models

__all__ = ['lin_eval_model']
    

class LinEvalModel(nn.Module):
    """
    Build a Barlow Twins model.
    """
    def __init__(self, ssl_model):
        super(LinEvalModel, self).__init__()
        self.ssl_model = ssl_model

        for name, param in self.ssl_model.named_parameters():
            if name not in ['classifier.weight', 'classifier.bias']:
                param.requires_grad = False
        
        self.ssl_model.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.ssl_model.classifier.bias.data.zero_()

        self.sssl_model.eval()

    def forward(self, x):
        z = self.ssl_model.encoder(x)
        y_hat = self.ssl_model.classifier(z)
        return y_hat

def lin_eval_model(ssl_model):
    return LinEvalModel(ssl_model)