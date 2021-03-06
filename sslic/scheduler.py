import torch
import math

__all__ = ['get_scheduler']

from typing import List


def get_scheduler(name, *args, **kwargs):
    if name is None:
        return Scheduler(*args, **kwargs)
    elif name in ['simclr', 'barlow_twins', 'byol', 'vicreg', 'nnclr']:
        return warmup_cosine(*args, warmup_epochs=10, **kwargs)
    elif name == "ressl":
        return warmup_cosine(*args, warmup_epochs=5, **kwargs)
    elif name in ['simsiam', 'mocov2', 'twist', 'cosine']:
        return cosine(*args, **kwargs)
    elif name in ['cosine', 'warmup_cosine']:
        return globals()[name](*args, **kwargs)
    elif name == 'multi_step':
        return multi_step(*args, **kwargs)
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
    def current_lrs(self) -> List[float]:
        return [pg['lr'] for pg in self.optimizer.param_groups]

    @property
    def current_unfixed_lrs(self):
        unfixed_lrs = [
            pg['lr'] for pg in self.optimizer.param_groups if 'fix_lr' not in pg or not pg['fix_lr']
        ]
        return unfixed_lrs

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
            if not ('fix_lr' in param_group and param_group['fix_lr']):
                param_group['lr'] = next_lr

    def step(self, iteration=None) -> None:
        self.iter_count = iteration if iteration else self.iter_count
        if self.iter_count > self.iterations and self.verbose:
            raise ValueError("Your scheduler overstepped the maximum number of iterations")
        self.on_step()
        self.iter_count += 1

    def __repr__(self):
        return f"Scheduler: {type(self).__name__} with {type(self.optimizer).__name__}" + \
               f"(lr={self.current_lrs[0]:2.4f})"


class MultiStep(Scheduler):

    def __init__(self, *args, steps=[0.6, 0.8], scale=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps = sorted(steps)
        self.scale = scale
        unfixed_lrs = self.current_unfixed_lrs
        assert len(unfixed_lrs) > 0, "No learning rate to schedule"
        if len(unfixed_lrs) > 1:
            assert unfixed_lrs[1:] == unfixed_lrs[:-1], "Unfixed learning rates must be the same"
        self.init_lr = unfixed_lrs[0]

    def on_step(self):
        if len(self.steps) and self.progress >= self.steps[0] and len(self.current_unfixed_lrs) > 0:
            self.steps.pop(0)
            next_lr = self.scale * self.current_unfixed_lrs[0]
            self.set_lr(next_lr)

    def __repr__(self):
        return f"Scheduler: {type(self).__name__} with {type(self.optimizer).__name__}" + \
               f"(init_lr={self.init_lr:2.4f}, step={self.steps}, scale={self.scale})"


class CosineAnnealing(Scheduler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        unfixed_lrs = self.current_unfixed_lrs
        assert len(unfixed_lrs) > 0, "No learning rate to schedule"
        if len(unfixed_lrs) > 1:
            assert unfixed_lrs[1:] == unfixed_lrs[:-1], "Unfixed learning rates must be the same"
        self.init_lr = unfixed_lrs[0]

    def on_step(self):
        next_lr = self.init_lr * 0.5 * (1. + math.cos(math.pi * self.progress))
        self.set_lr(next_lr)

    def __repr__(self):
        return f"Scheduler: {type(self).__name__} with {type(self.optimizer).__name__}" + \
               f"(init_lr={self.init_lr:2.4f})"


class WarmUpCosineAnnealing(CosineAnnealing):

    def __init__(self, *args, warmup_epochs=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmup_epochs = warmup_epochs
        self.warmup_iters = warmup_epochs * self.ipe
        assert self.warmup_iters < self.iterations

    @property
    def cos_progress(self) -> float:
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
            next_lr = self.init_lr * self.warmup_progress
            self.set_lr(next_lr)
        else:
            next_lr = self.init_lr * 0.5 * (1. + math.cos(math.pi * self.cos_progress))
            self.set_lr(next_lr)

    def __repr__(self):
        return f"Scheduler: {type(self).__name__} with {type(self.optimizer).__name__}" + \
               f"(max_lr={self.init_lr:2.4f}, warmup={self.warmup_epochs:2d})"


def cosine(*args, **kwargs):
    return CosineAnnealing(*args, **kwargs)


def warmup_cosine(*args, **kwargs):
    return WarmUpCosineAnnealing(*args, **kwargs)


def multi_step(*args, **kwargs):
    return MultiStep(*args, **kwargs)