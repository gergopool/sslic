import torch
from torch import nn
from torch.utils.data import DataLoader
import pkbar
import os
from abc import ABC

from typing import Tuple

from ..utils import WarmupCosineSchedule


class GeneralTrainer(ABC):
    """GeneralTrainer
    A trainer class responsible for training N epochs.
    This is an abstract class, both SSL and LinearEval methods will
    be inherited from this class.

    Parameters
    ----------
    model : nn.Module
        The model we desire to train.
    optimizer : torch.optim.Optimizer
        The optimizer to use for training.
    data_loaders : Tuple[DataLoader, DataLoader]
        Train and validation data loaders
    rank : int, optional
        The rank of this process. If None, it will assume the
        training is running a single process. By default None
    save_params : dict, optional
        Parameters used for saving the network.
        These parameters can be the name of the method and the name
        of the dataset which together define the model architecture
        the code built up. It must also contain a save_dir key
        which tells to which folder the model should be saved. If
        save_dir is None, the trainer will not save anything.
        By default {"save_dir": None}
    """

    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 data_loaders: Tuple[DataLoader, DataLoader],
                 rank: int = None,
                 save_params: dict = {"save_dir": None}):
        self.model = model
        self.optimizer = optimizer
        self.train_loader, self.val_loader = data_loaders
        self.rank = rank
        self.save_dir = save_params.pop('save_dir')
        self.save_dict = save_params
        self.scaler = torch.cuda.amp.GradScaler()

        # Progress bar with running average metrics
        self.pbar = ProgressBar(data_loaders, rank)

        # Checkpoints in which we save the model
        self.save_checkpoints = []

    def _need_save(self, epoch: int) -> bool:
        """_need_save
        Determines if the model should be saved in this
        particular epoch.
        """
        save_dir_given = self.save_dir is not None
        in_saving_epoch = (epoch + 1) in self.save_checkpoints
        is_saving_core = self.rank is None or self.rank == 0
        return save_dir_given and in_saving_epoch and is_saving_core

    def _ckp_name(self, epoch: int):
        """_ckp_name
        Checkpoint filename. It has an own unique class, because
        each trainer might define different filenames.
        """
        return f'checkpoint_{epoch+1:04d}.pth.tar'

    def _save(self, epoch: int):
        """_save [summary]
        Save the current checkpoint to a file.
        """
        save_dict = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'amp': self.scaler.state_dict(),
        }
        save_dict.update(self.save_dict)
        filename = self._ckp_name(epoch)
        os.makedirs(self.save_dir, exist_ok=True)
        filepath = os.path.join(self.save_dir, filename)
        torch.save(save_dict, filepath)

    def train(self, n_epochs: int, ref_lr: float = 0.1, n_warmup_epochs: int = 10):
        """train
        Train n epochs.

        Parameters
        ----------
        n_epochs : int
            Number of epochs to train.
        ref_lr : float, optional
            Base learning rate to cosine scheduler, by default 0.1
        n_warmup_epochs : int, optional
            Number of warmup epochs, by default 10
        """

        n_warmup_iter = len(self.train_loader) * n_warmup_epochs
        self.scheduler = WarmupCosineSchedule(optimizer=self.optimizer,
                                              warmup_steps=n_warmup_iter,
                                              T_max=n_epochs,
                                              ref_lr=ref_lr)

        for epoch in range(n_epochs):
            # Reset progress bar to the start of the line
            self.pbar.reset(epoch, n_epochs)

            # Train one epoch
            for data_batch in self.train_loader:
                metrics = self.train_step(data_batch)
                self.pbar.update(metrics)

            # Validate one epoch
            for data_batch in self.val_loader:
                metrics = self.val_step(data_batch)
                self.pbar.update(metrics)

            # Save network
            if self._need_save(epoch):
                self._save(epoch)

    def train_step(self, batch: torch.Tensor):
        raise NotImplementedError

    def val_step(self, batch: torch.Tensor):
        raise NotImplementedError

    def _accuracy(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """_accuracy 
        Accuracy of the model
        """
        pred = torch.max(y_hat.data, 1)[1]
        acc = (pred == y).sum() / len(y)
        return acc


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
            values = [(k, v) for (k, v) in value_dict.items()]
            self.kbar.add(1, values=values)