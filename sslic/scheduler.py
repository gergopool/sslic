import torch
import math

__all__ = ['get_scheduler']


def get_scheduler(name, *args, **kwargs):
    if name is None:
        return Scheduler(*args, **kwargs)
    elif name in ['simclr', 'barlow_twins']:
        return warmup_cosine(*args, warmup_epochs=10, **kwargs)
    elif name == "ressl":
        return warmup_cosine(*args, warmup_epochs=5, **kwargs)
    elif name == "simsiam":
        return cosine(*args, **kwargs)
    elif name in ['cosine', 'warmup_cosine']:
        return globals()[name](*args, **kwargs)
    else:
        raise NameError(f"Unknown scheduler: {name}")


class Scheduler:

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 epochs: int,
                 ipe: int,
                 verbose: bool = False):
        self.optimizer = optimizer
        self.epochs = epochs
        self.ipe = ipe
        self.verbose = verbose
        self.iter_count = 0

    @property
    def iterations(self) -> int:
        return int(self.epochs * self.ipe)

    @property
    def current_lr(self) -> float:
        # TODO Lars?
        return self.optimizer.param_groups[0]['lr']

    @property
    def progress(self) -> float:
        return self.iter_count / self.iterations

    @property
    def is_epoch_end(self) -> bool:
        return self.iter_count % self.ipe == 0

    def set_epoch(self, epoch):
        self.step(epoch * self.ipe)

    def on_step(self) -> None:
        pass

    def on_epoch_end(self) -> None:
        # Example of a possible feature later
        raise NotImplementedError

    def set_lr(self, next_lr) -> None:
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = next_lr

    def step(self, iteration=None) -> None:
        self.iter_count = iteration if iteration else self.iter_count + 1
        if self.iter_count >= self.iterations and self.rank == 0:
            raise ValueError("Your scheduler overstepped the maximum number of iterations")
        self.on_step()


class CosineAnnealing(Scheduler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_lr = self.current_lr

    def set_lr(self, next_lr: float) -> None:
        for param_group in self.optimizer.param_groups:
            if 'fix_lr' in param_group and param_group['fix_lr']:
                param_group['lr'] = self.init_lr
            else:
                param_group['lr'] = next_lr

    def on_step(self):
        next_lr = self.init_lr * 0.5 * (1. + math.cos(math.pi * self.progress))
        self.set_lr(next_lr)


class WarmUpCosineAnnealing(CosineAnnealing):

    def __init__(self, *args, warmup_epochs=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmup_epochs = warmup_epochs
        self.warmup_iters = warmup_epochs * self.ipe
        assert self.warmup_iters < self.iterations

    @property
    def progress(self) -> float:
        # Note: if negative, we're in warm up
        return (self.iter_count - self.warmup_iters) / (self.iterations - self.warmup_iters)

    @property
    def warmup_progress(self) -> float:
        # Note: if > 1, we're out of warm up
        return self.iter_count / self.warmup_iters

    @property
    def in_warmup(self):
        return self.warmup_progress < 1

    def on_step(self):
        if self.in_warmup:
            next_lr = self.init_lr * 0.5 * (1. + math.cos(math.pi * self.progress))
            self.set_lr(next_lr)
        else:
            super().on_step()


def cosine(*args, **kwargs):
    return CosineAnnealing(*args, **kwargs)


def warmup_cosine(*args, **kwargs):
    return WarmUpCosineAnnealing(*args, **kwargs)