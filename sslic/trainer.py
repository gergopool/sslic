import torch
import pkbar
from abc import ABC


class GeneralTrainer(ABC):
    def __init__(self, model, optimizer, data_loaders, device):
        self.model = model
        self.optimizer = optimizer
        self.train_loader, self.val_loader = data_loaders
        self.device = device
        self.model.to(device)

    def train(self, n_epochs):

        for epoch in range(n_epochs):
            kbar = self._get_kbar(epoch, n_epochs)
            for data_batch in self.train_loader:
                loss, acc = self.train_step(data_batch)
                kbar.add(1, values=[("loss", loss), ("acc", acc)])
            for data_batch in self.val_loader:
                loss, acc = self.val_step(data_batch)
                kbar.add(1, values=[("val_loss", loss), ("val_acc", acc)])

    def train_step(self, batch):
        raise NotImplementedError

    def val_step(self, batch):
        (x, y) = batch
        self.model.eval()

        y = y.to(self.device)
        x = x.to(self.device)

        with torch.no_grad():
            y_hat = self.model.classifier(self.model.encoder(x))
            loss = self.model.classifier_loss(y_hat, y)

        return loss.item(), self._accuracy(y_hat, y)

    def _accuracy(self, y_hat, y):
        pred = torch.max(y_hat.data, 1)[1]
        acc = (pred == y).sum() / len(y)
        return acc

    def _get_kbar(self, epoch_i, n_epochs):
        n = len(self.train_loader) + len(self.val_loader)
        return pkbar.Kbar(target=n,
                          epoch=epoch_i,
                          num_epochs=n_epochs,
                          width=8,
                          always_stateful=False)


class SSLTrainer(GeneralTrainer):
    def train_step(self, batch):
        (x, y) = batch
        self.model.train()

        y = y.to(self.device)
        x = [t.to(self.device) for t in x]
        y_hat, representations = self.model(x)

        # Linear layer loss
        # Note: this is safe to do because the representations do not
        # recieive gradients from the labels, the linear layer is detached
        cls_loss = self.model.classifier_loss(y_hat, y)
        ssl_loss = self.model.ssl_loss(*representations)
        loss = ssl_loss + cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), self._accuracy(y_hat, y)


class LinearEvalTrainer(GeneralTrainer):
    def __init__(self, model, optimizer, n_classes):
        super(SSLTrainer, self).__init__(model, optimizer)

    def train_step(self, batch):
        (x, y) = batch
        y = y.to(self.device)
        x = x = x.to(self.device)
        self.model.encoder.eval()

        # Avoiding unnecessary computation
        # by only calling encoder + linear classifier
        z = self.model.encoder(x).detach()
        y_hat = self.model.classifier(z)
        loss = self.model.classifier_loss(y_hat, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), self._accuracy(y_hat, y)
