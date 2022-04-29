import torch
from abc import ABC
from typing import Tuple, Dict

from .general import GeneralTrainer
from ..utils import AllReduce


class LinearEvalTrainer(GeneralTrainer):
    """LinearEvalTrainer

    Trainer for linear evaluation.
    """

    def __init__(self, *args, **kwargs):
        super(LinearEvalTrainer, self).__init__(*args, **kwargs)
        # Checkpoints in which we save
        self.save_checkpoints = [10, 50, 100]

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
        y = y.cuda()
        x = x.cuda()

        # Put model to eval mode to avoid BatchNorm updates
        self.model.eval()

        # Remove all possible gradients
        self.optimizer.zero_grad()

        # Predict
        y_hat = self.model(x)
        loss = self.model.ssl_loss(y_hat, y)

        # Backprop
        loss.backward()
        self.optimizer.step()

        # Get accuracy among all processes
        acc = self._accuracy(y_hat, y)
        return {'loss': loss.item(), 'acc': acc}

    def val_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """val_step

        A single val step, with a forward pass and metrics calculation.

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
        y = y.cuda()
        x = x.cuda()

        # Put model to eval mode to avoid BatchNorm updates
        self.model.eval()

        # Predict
        with torch.no_grad():
            y_hat = self.model(x)

        # Calculate loss and metrics
        loss = AllReduce.apply(self.model.classifier_loss(y_hat, y))
        acc = AllReduce.apply(self._accuracy(y_hat, y))

        return {'loss': loss.item(), 'acc': acc}
