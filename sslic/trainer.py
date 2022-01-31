import torch
import pkbar
from abc import ABC

from .utils import WarmupCosineSchedule, AllReduce
from .evaluator import SnnEvaluator


class GeneralTrainer(ABC):
    def __init__(self, model, optimizer, data_loaders, device, rank=None):
        self.model = model
        self.optimizer = optimizer
        self.train_loader, self.val_loader = data_loaders
        self.device = device
        self.model = model
        self.scaler = torch.cuda.amp.GradScaler()
        self.rank = rank
        self.pbar = ProgressBar(data_loaders, rank)
        self.evaluator = SnnEvaluator(self.model.prev_dim, self.model.n_classes,
                                      5000 // self.model.n_classes).cuda()

    def train(self, n_epochs, ref_lr=0.1):

        self.scheduler = WarmupCosineSchedule(optimizer=self.optimizer,
                                              warmup_steps=len(self.train_loader) * 10,
                                              T_max=n_epochs,
                                              ref_lr=ref_lr)

        for epoch in range(n_epochs):
            self.pbar.reset(epoch, n_epochs)
            for data_batch in self.train_loader:
                loss, lin_acc, snn_acc = self.train_step(data_batch)
                self.pbar.update([("loss", loss), ('lin_acc', lin_acc), ('snn_acc', snn_acc)])
            for data_batch in self.val_loader:
                lin_acc, snn_acc = self.val_step(data_batch)
                self.pbar.update([("val_lin_acc", lin_acc), ("val_snn_acc", snn_acc)])

    def train_step(self, batch):
        raise NotImplementedError

    def val_step(self, batch):
        (x, y) = batch
        self.model.eval()

        y = y.cuda()
        x = x.cuda()

        with torch.no_grad():
            z = self.model.encoder(x)
            y_hat = self.model.classifier(z)

            snn_top1 = AllReduce.apply(self.evaluator(z, y))[0]
            lin_top1 = AllReduce.apply(self._accuracy(y_hat, y))
            
        return lin_top1, snn_top1

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

        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):

            y = y.cuda(non_blocking=True)
            x = [t.cuda(non_blocking=True) for t in x]
            cnn_out, representations = self.model(x)
            y_hat = self.model.classifier(cnn_out)

            snn1_acc = AllReduce.apply(self.evaluator(cnn_out, y))[0]
            lin1_acc = AllReduce.apply(self._accuracy(y_hat, y))
            self.evaluator.update(cnn_out, y)

            # Linear layer loss
            # Note: this is safe to do because the representations do not
            # recieive gradients from the labels, the linear layer is detached
            cls_loss = self.model.classifier_loss(y_hat, y)
            ssl_loss = self.model.ssl_loss(*representations)
            loss = ssl_loss + cls_loss

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return ssl_loss.item(), lin1_acc, snn1_acc


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


class ProgressBar:
    def __init__(self, data_loaders, rank):
        self.n_iter = len(data_loaders[0]) + len(data_loaders[1])
        self.kbar = None
        self.is_active = rank is None or rank == 0

    def reset(self, epoch_i, n_epochs):
        if self.is_active:
            self.kbar = pkbar.Kbar(target=self.n_iter,
                                   epoch=epoch_i,
                                   num_epochs=n_epochs,
                                   width=8,
                                   always_stateful=False)

    def update(self, value_list):
        if self.is_active:
            self.kbar.add(1, values=value_list)