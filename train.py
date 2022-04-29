import argparse
import os
import torch
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from random import randint
import torch.backends.cudnn as cudnn

from sslic.logger import Logger
from sslic.scheduler import get_scheduler
from sslic.trainers import SSLTrainer
from sslic.models import get_ssl_network
from sslic.data import get_dataset
import sslic.utils as utils
from sslic.optimizers import get_optimizer
from sslic.losses import get_loss
from ssl_eval import Evaluator

METHODS = [
    'simsiam', 'simclr', 'barlow_twins', 'ressl', 'vicreg', 'twist', 'mocov2', 'byol', 'nnclr'
]
DATASETS = ['imagenet', 'tiny_imagenet', 'cifar10', 'cifar100']

parser = argparse.ArgumentParser(description='Simple settings.')
parser.add_argument('method', type=str, choices=METHODS)
parser.add_argument('data_root', type=str)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--dataset', type=str, default='cifar10', choices=DATASETS)
parser.add_argument('--run-name', type=str, default='dev')
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--loss', type=str, default=None)
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--opt', type=str, default=None)
parser.add_argument('--transform', type=str, default=None)
parser.add_argument('--scheduler', type=str, default=None)
parser.add_argument('--save-dir', type=str, default="checkpoints")
parser.add_argument('--devices', type=str, nargs='+', default=[])


def get_data_loaders(rank, world_size, per_gpu_batch_size, args):
    '''Define data loaders to a specific process.'''

    # Create datasets
    method = args.transform if args.transform else args.method
    train_dataset = get_dataset(args.data_root, method, args.dataset, 'ssl')
    val_dataset = get_dataset(args.data_root, method, args.dataset, 'test')

    # Create distributed samplers if multiple processes defined
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    # Data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=per_gpu_batch_size,
                                               shuffle=shuffle,
                                               num_workers=8,
                                               pin_memory=True,
                                               drop_last=True,
                                               persistent_workers=True,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=per_gpu_batch_size,
                                             shuffle=False,
                                             num_workers=8,
                                             pin_memory=True,
                                             persistent_workers=True,
                                             sampler=val_sampler)
    return train_loader, val_loader


def get_model(world_size, args):
    # Define model
    kwargs = {}
    if args.loss:
        kwargs['ssl_loss'] = get_loss(args.loss)
    model = get_ssl_network(args.method, args.dataset, **kwargs).cuda()

    # Create distributed version if needed
    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = utils.DDP(model)

    return model


def main(rank, world_size, port, args):

    # Set device and distributed settings
    if torch.cuda.is_available():
        device = torch.cuda.device(rank)
        torch.cuda.set_device(device)

    world_size, rank = utils.init_distributed(port, rank_and_world_size=(rank, world_size))
    if world_size > 1:
        print(f"Rank{rank} started succesfully.")
        torch.distributed.barrier()

    # Divide batch size
    per_gpu_batch_size = args.batch_size // world_size

    # Data Loaders
    train_loader, val_loader = get_data_loaders(rank, world_size, per_gpu_batch_size, args)

    # Model
    model = get_model(world_size, args)

    method = args.opt if args.opt else args.method
    optimizer = get_optimizer(method, model, batch_size=args.batch_size, lr=args.lr)

    # Training parameters
    save_dir = os.path.join(args.save_dir, f"{args.method}_{args.dataset}", args.run_name)
    save_params = {"method": args.method, "dataset": args.dataset, "save_dir": save_dir}

    # Evaluator
    n_views = 2 if args.dataset == 'imagenet' else 4
    evaluator = Evaluator(model.encoder,
                          args.dataset,
                          args.data_root,
                          n_views=n_views,
                          batch_size=per_gpu_batch_size)

    # Logger
    logger = Logger(log_dir=save_dir,
                    global_step=0,
                    batch_size=args.batch_size,
                    world_size=world_size,
                    log_per_sample=1e4)
    logger.log_config(vars(args))

    # Scheduler
    scheduler_name = args.scheduler if args.scheduler else args.method
    scheduler = get_scheduler(scheduler_name,
                              optimizer=optimizer,
                              epochs=args.epochs,
                              ipe=len(train_loader),
                              verbose=rank == 0)

    if rank == 0:
        print(model)
        print(scheduler)

    trainer = SSLTrainer(model,
                         scheduler, (train_loader, val_loader),
                         save_params=save_params,
                         evaluator=evaluator,
                         logger=logger)

    cudnn.benchmark = True

    # Load if needed
    if args.resume:
        trainer.load(args.resume)

    # Train
    trainer.train(args.epochs)


if __name__ == '__main__':
    args = parser.parse_args()

    if args.devices:
        str_devices = ','.join(args.devices)
        os.environ['CUDA_VISIBLE_DEVICES'] = str_devices

    num_gpus = torch.cuda.device_count()

    # Choose a random port so multiple runs won't conflict with
    # a large chance.
    port = randint(0, 9999) + 40000

    if num_gpus > 1:
        try:
            mp.spawn(main, nprocs=num_gpus, args=(num_gpus, port, args))
        except KeyboardInterrupt:
            print('\nInterrupted. Attempting a graceful shutdown..')
    else:
        main(0, 1, port, args)
