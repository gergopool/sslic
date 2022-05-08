import torch

from typing import Tuple, Dict

from .general import GeneralTrainer


class SSLTrainer(GeneralTrainer):
    """ SSLTrainer

    Trainer for self-supervised image classification.
    """

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
        (x, _) = batch
        self.model.train()

        # Remove all possible gradients
        self.optimizer.zero_grad(set_to_none=True)

        # Use fp16 to save memory
        with torch.cuda.amp.autocast(enabled=True):

            # Predict
            representations = self.model(x)

            # For loss calculation use fp32
            with torch.cuda.amp.autocast(enabled=False):

                # Convert back to fp32
                representations = [
                    x.to(torch.float32, memory_format=torch.contiguous_format, non_blocking=True)
                    for x in representations
                ]

                # Calculate loss
                loss = self.model.ssl_loss(*representations)
                log_dict = {}

                # Log
                if self.logger.need_log():
                    for i, rep in enumerate(representations):
                        self.logger.log_describe(f"stats/representation_{i}_len",
                                                 rep.norm(dim=-1).detach())

                if type(loss) is tuple:
                    loss, log_dict = loss

        # Backprop
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {**log_dict, "loss": loss.item()}