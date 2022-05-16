import torch
from abc import ABC
from typing import Tuple, Dict

from .general import GeneralTrainer
from ..utils import AllGather, AllReduce


class LinearEvalTrainer(GeneralTrainer):
    """LinearEvalTrainer

    Trainer for linear evaluation.
    """

    def __init__(self, *args, **kwargs):
        super(LinearEvalTrainer, self).__init__(*args, **kwargs)
        # Checkpoints in which we save
        self.save_checkpoints = [10, 50, 100]
        self.eval_checkpoints = [1, 10, 20, 30, 40, 50, 100]
        self.half_precision = True

    def _ckp_name(self, epoch):
        """_ckp_name 
        Checkpoint name used for linear eval models.
        """
        return f'lin_eval_checkpoint_{epoch+1:04d}.pth.tar'

    def _accuracy(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """_accuracy 
        Accuracy of the model
        """
        pred = torch.max(y_hat.data, 1)[1]
        acc = (pred == y).sum() / len(y)
        return AllReduce.apply(acc)

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """train_step

        A single train step, including the forward and backward passes.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            The (x,y) pair provided by the generator.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary of metrics. E.g. loss, top1 accuracy, top5 accuracy
        """
        (x, y) = batch

        # Put model to eval mode to avoid BatchNorm updates
        # Note: only linear layer is trained which has no batchnorm
        self.model.eval()

        # Remove all possible gradients
        self.optimizer.zero_grad(set_to_none=True)

        # Predict
        with torch.cuda.amp.autocast(enabled=self.half_precision):
            y_hat = self.model(x)
            loss = self.model.criterion(y_hat, y)

        # Backprop
        # loss.backward()
        # self.optimizer.step()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Accuracy
        acc = self._accuracy(y_hat, y)
        return {'loss': loss.item(), 'acc': acc}

    def run_validation(self):

        # Note: Although we count moving average accuracy on progress bar, we
        # also count the total number of hits and validation points. This is
        # made for official results, e.g. data loaders with drop_last=False
        # will be counted correctly.
        hits, total = 0, 0

        for data_batch in self._iter_with_convert(self.val_loader, self.device):
            batch_hits, batch_n = self.val_step(data_batch)
            hits += batch_hits
            total += batch_n
        acc = (hits / total) * 100
        if self.rank == 0:
            print(f"Accuracy: {acc:3.2f}%")

    def val_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[int, int]:
        """val_step

        A single val step, with a forward pass and metrics calculation.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            The (x,y) pair provided by the generator.

        Returns
        -------
        Tuple[int, int]
            Number of hits and number of datapoints, respectively.
        """
        (x, y) = batch

        # Put model to eval mode to avoid BatchNorm updates
        self.model.eval()

        # Predict
        with torch.cuda.amp.autocast(enabled=self.half_precision):
            y_hat = self.model(x)

        # Calculate loss and metrics
        y_hat = AllGather.apply(y_hat)
        y = AllGather.apply(y)

        hits = (y_hat.argmax(dim=1) == y).sum()
        total = len(y)
        return int(hits), int(total)