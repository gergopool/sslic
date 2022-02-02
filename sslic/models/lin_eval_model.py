import torch.nn as nn

__all__ = ['lin_eval_model']
    

class LinEvalModel(nn.Module):
    """
    Build a Barlow Twins model.
    """
    def __init__(self, ssl_model):
        super(LinEvalModel, self).__init__()
        self.encoder = ssl_model.encoder
        self.classifier = ssl_model.classifier
        self.classifier_loss = ssl_model.classifier_loss

        for name, param in self.named_parameters():
            if name not in ['classifier.weight', 'classifier.bias']:
                param.requires_grad = False
        
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.classifier.bias.data.zero_()

    def forward(self, x):
        z = self.encoder(x)
        y_hat = self.classifier(z)
        return y_hat

def lin_eval_model(ssl_model):
    return LinEvalModel(ssl_model)