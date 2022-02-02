import torch
from abc import ABC

from typing import Tuple, Dict

from .general import GeneralTrainer
from ..utils import AllReduce
from ..evaluator import SnnEvaluator


class SSLTrainer(GeneralTrainer):
    """ SSLTrainer

    Trainer for self-supervised image classification.
    """

    def __init__(self, *args, **kwargs):
        super(SSLTrainer, self).__init__(*args, **kwargs)

        # Checkpoints in which we save
        self.save_checkpoints = [1, 10, 20, 50, 100, 200, 400, 600, 800, 1000]

        # Soft nearest neighbor evaluator
        self.evaluator = SnnEvaluator(self.model.prev_dim,
                                      self.model.n_classes,
                                      5000 // self.model.n_classes).cuda()

    def _ckp_name(self, epoch):
        """_ckp_name 
        Checkpoint name used for self-supervised pretrained models.
        """
        return f'ssl_checkpoint_{epoch+1:04d}.pth.tar'

    def train_step(self, batch: Tuple[Tuple[torch.Tensor],
                                      torch.Tensor]) -> Dict[str, torch.Tensor]:
        """train_step

        A single train step, including the forward and backward passes.

        Parameters
        ----------
        batch : Tuple[Tuple[torch.Tensor], torch.Tensor]
            The (x,y) pair provided by the generator. Note that ssl methods
            require multiple views of an image and therefore x is a tuple
            of image views.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary of metrics. E.g. loss, top1 accuracy, top5 accuracy
        """
        (x, y) = batch
        self.model.train()
        metrics = {}

        # Remove all possible gradients
        self.optimizer.zero_grad()

        # Use fp16 to save memory
        with torch.cuda.amp.autocast(enabled=True):

            # Predict
            y = y.cuda(non_blocking=True)
            x = [t.cuda(non_blocking=True) for t in x]
            cnn_out, representations = self.model(x)
            y_hat = self.model.classifier(cnn_out.detach())

        # For loss calculation use fp32
        with torch.cuda.amp.autocast(enabled=False):
            # Convert back to fp32
            y_hat = y_hat.float()
            representations = [x.float() for x in representations]

            # Calculate loss
            ssl_loss = self.model.ssl_loss(*representations)
            cls_loss = self.model.classifier_loss(y_hat, y)
            loss = ssl_loss + cls_loss

            # Metrics
            # Note: calculating approximate linear eval accuracy is safe
            # because the cnn embeddings are detached
            cnn_out = cnn_out.float()
            metrics['snn1_acc'] = AllReduce.apply(self.evaluator(cnn_out.detach(), y))
            metrics['lin1_acc'] = AllReduce.apply(self._accuracy(y_hat, y))
            self.evaluator.update(cnn_out, y)

        # Backprop
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        metrics['ssl_loss'] = ssl_loss.item()

        return metrics

    def val_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
        self.model.eval()
        metrics = {}

        # Move data to cuda
        y = y.cuda()
        x = x.cuda()

        # Since these metrics are nothing like official,
        # use fp16 to get an approximate values of desired metrics
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                z = self.model.encoder(x)
                y_hat = self.model.classifier(z)

                metrics['snn_top1'] = AllReduce.apply(self.evaluator(z, y))
                metrics['lin_top1'] = AllReduce.apply(self._accuracy(y_hat, y))

        return metrics
