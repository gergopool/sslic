import argparse
import os

os.environ['TORCH_DISTRIBUTED_DEBUG'] = "DETAIL"
import torch
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from random import randint
import torch.backends.cudnn as cudnn

from sslic.trainers import SSLTrainer
from sslic.models import get_ssl_network
from sslic.data import get_dataset_provider
import sslic.utils as utils
from sslic.optimizers import get_optimizer
from sslic.losses import get_loss
from ssl_eval import Evaluator

parser = argparse.ArgumentParser(description='Simple settings.')
parser.add_argument('method', type=str, choices=['simsiam', 'simclr', 'barlow_twins', 'ressl'])
parser.add_argument('data_root', type=str)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--run-name', type=str, default='dev')
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--loss', type=str, default=None)
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--opt', type=str, default=None)
parser.add_argument('--save_dir', type=str, default="checkpoints")
parser.add_argument('--devices', type=str, nargs='+', default=['0'])


def get_data_loaders(rank, world_size, per_gpu_batch_size, args):
    '''Define data loaders to a specific process.'''

    # Create datasets
    dataset_provider = get_dataset_provider(args.data_root, args.dataset, method_name=args.method)
    train_dataset = dataset_provider('ssl')
    val_dataset = dataset_provider('test')

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
        # model.ssl_loss = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.ssl_loss)
        # model.ssl_loss = utils.DDP(model.ssl_loss)

    return model


def main(rank, world_size, port, args):

    # Set device and distributed settings
    if torch.cuda.is_available():
        device = torch.cuda.device(int(args.devices[rank]))
        torch.cuda.set_device(device)
    world_size, rank = utils.init_distributed(port, rank_and_world_size=(rank, world_size))
    if world_size > 1:
        print(f"Rank{rank} running..")
        # torch.distributed.barrier()

    # Divide batch size
    per_gpu_batch_size = args.batch_size // world_size

    # Data Loaders
    train_loader, val_loader = get_data_loaders(rank, world_size, per_gpu_batch_size, args)

    # Model
    model = get_model(world_size, args)
    print(model)

    method = args.opt if args.opt else args.method
    optimizer = get_optimizer(method, model, batch_size=args.batch_size, lr=args.lr)

    # Training parameters
    save_dir = os.path.join(args.save_dir, f"{args.run_name}/{args.method}_{args.dataset}")
    save_params = {"method": args.method, "dataset": args.dataset, "save_dir": save_dir}

    evaluator = None
    evaluator = Evaluator(model.encoder,
                          args.dataset,
                          args.data_root,
                          n_views=1,
                          batch_size=per_gpu_batch_size)

    trainer = SSLTrainer(model,
                         optimizer, (train_loader, val_loader),
                         save_params=save_params,
                         evaluator=evaluator)

    cudnn.benchmark = True

    # Train
    trainer.train(args.epochs, ref_lr=optimizer.param_groups[0]['lr'])


if __name__ == '__main__':
    args = parser.parse_args()
    # torch.multiprocessing.set_start_method("forkserver")
    num_gpus = len(args.devices)

    # Choose a random port so multiple runs won't conflict with
    # a large chance.
    port = randint(0, 9999) + 40000

    if len(args.devices) > 1:
        mp.spawn(main, nprocs=num_gpus, args=(num_gpus, port, args))
    else:
        main(0, 1, port, args)
