import torch
import pkbar
import os
from abc import ABC

from .utils import WarmupCosineSchedule, AllReduce
from .evaluator import SnnEvaluator


class GeneralTrainer(ABC):
    def __init__(self, model, optimizer, data_loaders, device, rank=None, save_params={"save_dir":None}):
        self.model = model
        self.optimizer = optimizer
        self.train_loader, self.val_loader = data_loaders
        self.device = device
        self.model = model
        self.scaler = torch.cuda.amp.GradScaler()
        self.rank = rank
        self.save_dir = save_params.pop('save_dir')
        self.save_dict = save_params
        self.pbar = ProgressBar(data_loaders, rank)
        self.evaluator = SnnEvaluator(self.model.prev_dim, self.model.n_classes,
                                      5000 // self.model.n_classes).cuda()
        self.save_checkpoints = []
        
    def _need_save(self, epoch):
        save_dir_given = self.save_dir is not None
        in_saving_epoch = (epoch+1) in self.save_checkpoints
        is_saving_core =  self.rank is None or self.rank==0
        return save_dir_given and in_saving_epoch and is_saving_core

    def _save(self, epoch):
        save_dict = {
            'epoch': epoch+1,
            'state_dict' : self.mdoel.state_dict(),
            'optimizer' : self.optimzer.state_dict(),
            'amp' : self.scaler.state_dict(),
        }
        save_dict.update(self.save_dict)
        filename = f'checkpoint_{epoch:04d}.pt.tar'
        os.makedirs(self.save_dict, exist_ok=True)
        filepath = os.path.join(self.save_dir, filename)
        torch.save(save_dict, filepath)




    def train(self, n_epochs, ref_lr=0.1):

        self.scheduler = WarmupCosineSchedule(optimizer=self.optimizer,
                                              warmup_steps=len(self.train_loader) * 10,
                                              T_max=n_epochs,
                                              ref_lr=ref_lr)

        for epoch in range(n_epochs):
            self.pbar.reset(epoch, n_epochs)
            for data_batch in self.train_loader:
                metrics = self.train_step(data_batch)
                self.pbar.update(metrics)
            for data_batch in self.val_loader:
                metrics = self.val_step(data_batch)
                self.pbar.update(metrics)

            if self._need_save(epoch):
                self._save(epoch)

    def train_step(self, batch):
        raise NotImplementedError

    def val_step(self, batch):
        (x, y) = batch
        self.model.eval()

        metrics = {}

        y = y.cuda()
        x = x.cuda()

        with torch.no_grad():
            z = self.model.encoder(x)
            y_hat = self.model.classifier(z)

            metrics['snn_top1'] = AllReduce.apply(self.evaluator(z, y))
            metrics['lin_top1'] = AllReduce.apply(self._accuracy(y_hat, y))
            
        return metrics

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

    def __init__(self, *args, **kwargs):
        super(SSLTrainer, self).__init__(*args, **kwargs)
        self.save_checkpoints = [1, 10, 20, 50, 100, 200, 400, 600, 800, 1000]

    def train_step(self, batch):
        (x, y) = batch
        self.model.train()

        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            y = y.cuda(non_blocking=True)
            x = [t.cuda(non_blocking=True) for t in x]
            cnn_out, representations = self.model(x)
            y_hat = self.model.classifier(cnn_out)

            snn1_acc = AllReduce.apply(self.evaluator(cnn_out, y))
            lin1_acc = AllReduce.apply(self._accuracy(y_hat, y))
            self.evaluator.update(cnn_out, y)

            # Linear layer loss
            # Note: this is safe to do because the representations do not
            # recieive gradients from the labels, the linear layer is detached
        with torch.cuda.amp.autocast(enabled=False):
            y_hat = y_hat.float()
            representations = [x.float() for x in representations]
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

    def update(self, value_dict):
        if self.is_active:
            values = [(k,v) for (k,v) in value_dict.items()]
            self.kbar.add(1, values=values)