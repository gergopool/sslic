import torch
import torch.nn.functional as F

from .general import Loss

__all__ = ['byol_loss']


class BYOLLoss(Loss):
    """BYOLLoss Loss"""

    def mse(self, student_z: torch.Tensor, teacher_z: torch.Tensor) -> torch.Tensor:
        # This is equivalent to mse between normalized embs
        return 2 - 2 * F.cosine_similarity(student_z, teacher_z, dim=-1).mean()

    def forward(self,
                student_z1: torch.Tensor,
                student_z2: torch.Tensor,
                teacher_z1: torch.Tensor,
                teacher_z2: torch.Tensor) -> torch.Tensor:
        return self.mse(student_z1, teacher_z2) + self.mse(student_z2, teacher_z1)


def byol_loss() -> Loss:
    return BYOLLoss