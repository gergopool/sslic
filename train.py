import argparse
import torch
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from random import randint

from sslic.trainer import SSLTrainer
from sslic.models import get_ssl_method
from sslic.data import get_dataset_provider
import sslic.utils as utils
from sslic.lars import LARS

parser = argparse.ArgumentParser(description='Simple settings.')
parser.add_argument('method', type=str, choices=['simsiam', 'simclr', 'barlow_twins'])
parser.add_argument('data_root', type=str)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.3)
parser.add_argument('--devices', type=str, nargs='+', default=['0'])


def get_data_loaders(rank, world_size, args):
    dataset_provider = get_dataset_provider(args.data_root, args.method, args.dataset)

    train_dataset = dataset_provider('ssl')
    val_dataset = dataset_provider('test')

    train_sampler = None
    val_sampler = None
    shuffle = True
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        shuffle = False
        

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=shuffle,
                                               num_workers=8,
                                               pin_memory=True,
                                               persistent_workers=True,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=8,
                                             pin_memory=True,
                                             persistent_workers=True,
                                             sampler=val_sampler)
    return train_loader, val_loader

def get_model(rank, world_size, device, args):
    model = get_ssl_method(args.method, args.dataset).cuda()

    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = utils.DDP(model, broadcast_buffers=False)

    return model

def main(rank, world_size, port, args):
    if torch.cuda.is_available():
        device = torch.cuda.device(int(args.devices[rank]))
        torch.cuda.set_device(device)


    world_size, rank = utils.init_distributed(port, rank_and_world_size=(rank, world_size))

    # Data Loader
    train_loader, val_loader = get_data_loaders(rank, world_size, args)

    # Model
    model = get_model(rank, world_size, device, args)

    # Optimizer
    optimizer = LARS(model.parameters(), lr=args.lr, weight_decay=1e-4)

    trainer = SSLTrainer(model, optimizer, (train_loader, val_loader), device=device, rank=rank)
    trainer.train(args.epochs, args.lr)


if __name__ == '__main__':
    args = parser.parse_args()
    torch.multiprocessing.set_start_method("spawn")
    num_gpus = len(args.devices)
    port = randint(0, 9999) + 40000

    if len(args.devices) > 1:
        mp.spawn(main, nprocs=num_gpus, args=(num_gpus, port, args))
    else:
        main(0, 1, port, args)
