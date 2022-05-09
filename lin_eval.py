import argparse
import torch
import os
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from random import randint
import torch.backends.cudnn as cudnn

from sslic.trainers import LinearEvalTrainer
from sslic.models import get_lin_eval_network
from sslic.scheduler import get_scheduler
from sslic.optimizers import get_optimizer
from sslic.data import get_dataset
from sslic.data.transforms import get_transform
import sslic.utils as utils

OPT = "linear_eval"  # sgd
SCHEDULER = 'linear_eval'
TRANS = 'mocov2'

parser = argparse.ArgumentParser(description='Simple settings.')
parser.add_argument('pretrained', type=str)
parser.add_argument('data_root', type=str)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--devices', type=str, nargs='+', default=[])
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--n-workers', type=int, default=16)


def get_data_loaders(rank, world_size, per_gpu_batch_size, checkpoint, args):
    '''Define data loaders to a specific process.'''
    # Create datasets

    pretrain_dataset = checkpoint['dataset']
    train_trans = get_transform(TRANS, pretrain_dataset, split='train', norm=args.dataset)
    val_trans = get_transform(TRANS, pretrain_dataset, split='test', norm=args.dataset)
    train_dataset = get_dataset(args.data_root, args.dataset, train_trans, is_train=True)
    val_dataset = get_dataset(args.data_root, args.dataset, val_trans, is_train=False)

    # Create distributed samplers if multiple processes defined
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=per_gpu_batch_size,
                                               shuffle=shuffle,
                                               num_workers=args.n_workers,
                                               pin_memory=True,
                                               persistent_workers=True,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=per_gpu_batch_size,
                                             shuffle=False,
                                             num_workers=args.n_workers,
                                             pin_memory=True,
                                             persistent_workers=True,
                                             sampler=val_sampler)
    return train_loader, val_loader


def get_model(args, world_size, checkpoint):

    n_classes = {'imagenet': 1000, 'tiny_imagenet': 200, 'cifar10': 10, 'cifar100': 100}
    assert args.dataset in n_classes, f"Unknown dataset {args.dataset}"
    n_classes = n_classes[args.dataset]

    # Get params
    pretrain_dataset = checkpoint['dataset']
    old_state_dict = checkpoint['state_dict']
    new_state_dict = {}

    for k, v in old_state_dict.items():
        for prefix in ['module.encoder.', 'encoder.']:
            if k.startswith(prefix):
                new_state_dict[k[len(prefix):]] = v
    del old_state_dict

    # Define and load model
    # Pay attention for transfer learning opportunites, e.g. trained on
    # imagenet but evaluated on cifar10
    model = get_lin_eval_network(pretrain_dataset, n_classes=n_classes)
    memory_format = torch.channels_last if model.sync_batchnorm else torch.contiguous_format
    model = model.to(device='cuda', memory_format=memory_format)
    model.encoder.load_state_dict(new_state_dict, strict=True)

    # Create distributed version if needed
    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = utils.DDP(model, broadcast_buffers=False)

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

    # Get checkpoint
    checkpoint = torch.load(args.pretrained, map_location="cpu")

    # Data Loaders
    train_loader, val_loader = get_data_loaders(rank, world_size, per_gpu_batch_size, checkpoint, args)

    # Model
    model = get_model(args, world_size, checkpoint=checkpoint)

    # Scheduler
    optimizer = get_optimizer(OPT, model, batch_size=args.batch_size)
    scheduler = get_scheduler(SCHEDULER,
                              optimizer=optimizer,
                              epochs=args.epochs,
                              ipe=len(train_loader),
                              verbose=rank == 0)

    # Training parameters
    save_dir = os.path.split(args.pretrained)[0]
    save_dir = os.path.join(save_dir, args.dataset + "_eval")
    os.makedirs(save_dir, exist_ok=True)
    save_params = {"method": "linear_eval", "dataset": args.dataset, "save_dir": save_dir}

    trainer = LinearEvalTrainer(model,
                                scheduler, (train_loader, val_loader),
                                save_params=save_params)

    cudnn.benchmark = True

    if rank == 0:
        print(model)
        print(scheduler)

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
