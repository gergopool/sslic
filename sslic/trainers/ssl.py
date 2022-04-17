import torch

from typing import Tuple, Dict

from .general import GeneralTrainer


class SSLTrainer(GeneralTrainer):
    """ SSLTrainer

    Trainer for self-supervised image classification.
    """

    def __init__(self, *args, **kwargs):
        super(SSLTrainer, self).__init__(*args, **kwargs)

        # Checkpoints in which we save
        self.save_checkpoints = [1, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 800, 1000]

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
        device = self.device

        # Remove all possible gradients
        self.optimizer.zero_grad()

        # Use fp16 to save memory
        with torch.cuda.amp.autocast(enabled=True):

            # Predict
            y = y.to(device)
            x = [t.to(device) for t in x]
            representations = self.model(x)

            # For loss calculation use fp32
            with torch.cuda.amp.autocast(enabled=False):

                # Convert back to fp32
                representations = [x.float() for x in representations]

                # Calculate loss
                loss = self.model.ssl_loss(*representations)

        # Backprop
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # loss.backward()
        # self.optimizer.step()

        return {"loss": loss.item()}