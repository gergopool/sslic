import os
import math
import torch
import torch.distributed as dist


def after_init_world_size_n_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    else:
        return 1, 0


# Code credits:
# https://github.com/facebookresearch/suncet/blob/main/src/utils.py
def init_distributed(port=40011, rank_and_world_size=(None, None)):

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = rank_and_world_size
    os.environ['MASTER_ADDR'] = 'localhost'

    if (rank is None) or (world_size is None):
        try:
            world_size = int(os.environ['SLURM_NTASKS'])
            rank = int(os.environ['SLURM_PROCID'])
            os.environ['MASTER_ADDR'] = os.environ['HOSTNAME']
        except Exception:
            print('WARNING: Distributed training not available')
            world_size, rank = 1, 0
            return world_size, rank

    try:
        # Open a random port
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    except Exception:
        world_size, rank = 1, 0
        print('WARNING: Distributed training not available')

    return world_size, rank


class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1)):
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if (dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1)):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1)):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


class DDP(torch.nn.parallel.DistributedDataParallel):

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class WarmupCosineSchedule(torch.optim.lr_scheduler.LambdaLR):

    def __init__(self, optimizer, warmup_steps, T_max, ref_lr):
        self.warmup_steps = max(1, warmup_steps)
        self.T_max = max(1, T_max - warmup_steps)
        self.start_lr = ref_lr / 10
        self.final_lr = ref_lr / 10
        self.ref_lr = ref_lr
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=-1)

    def _warmup(self, step):
        progress = step / self.warmup_steps
        new_lr = progress * self.ref_lr + (1 - progress) * self.start_lr
        return new_lr / self.ref_lr

    def _cosine(self, step):
        progress = (step - self.warmup_steps) / self.T_max
        cos_progress = (1. + math.cos(math.pi * progress)) * 0.5
        new_lr = cos_progress * self.ref_lr + (1 - cos_progress) * self.final_lr
        return new_lr / self.ref_lr

    def lr_lambda(self, step):
        in_warump_phase = step < self.warmup_steps
        return self._warmup(step) if in_warump_phase else self._cosine(step)
