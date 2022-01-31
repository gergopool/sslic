import torch.nn as nn

class BaseModel(nn.Module):

    def __init__(self,  base_encoder, dim=128, ssl_loss=None, n_classes=1000, **kwargs):
        super(BaseModel, self).__init__()
        self.dim = dim
        self.ssl_loss = ssl_loss
        self.n_classes = n_classes

        # create the encoder
        self.encoder = base_encoder(**kwargs)
        self.prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Identity()

        self.classifier = nn.Linear(self.prev_dim, n_classes)
        self.lin_eval_loss = nn.CrossEntropyLoss()

    def classifier_loss(self, y_hat, y):
        return self.lin_eval_loss(y_hat, y)
