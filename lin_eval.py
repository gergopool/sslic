import argparse
import torch
import os
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from random import randint

from sslic.trainers import LinearEvalTrainer
from sslic.models import get_ssl_method, lin_eval_model
from sslic.data import get_dataset_provider
import sslic.utils as utils
from sslic.lars import LARS

parser = argparse.ArgumentParser(description='Simple settings.')
parser.add_argument('pretrained', type=str)
parser.add_argument('data_root', type=str)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--devices', type=str, nargs='+', default=['0'])


def get_data_loaders(rank, world_size, args):
    '''Define data loaders to a specific process.'''

    # Create datasets
    dataset_provider = get_dataset_provider(args.data_root, args.dataset)
    train_dataset = dataset_provider('train')
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


def get_model(world_size, checkpoint_params):

    # Get params
    method = checkpoint_params['method']
    dataset = checkpoint_params['dataset']
    state_dict = checkpoint_params['state_dict']

    # Define and load model
    ssl_model = get_ssl_method(method, dataset)
    ssl_model.load_state_dict(state_dict, strict=False)
    model = lin_eval_model(ssl_model).cuda()

    # Create distributed version if needed
    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = utils.DDP(model, broadcast_buffers=False)

    return model


def main(rank, world_size, port, args):

    # Set device and distributed settings
    if torch.cuda.is_available():
        device = torch.cuda.device(int(args.devices[rank]))
        torch.cuda.set_device(device)
    world_size, rank = utils.init_distributed(port, rank_and_world_size=(rank, world_size))

    # Load params
    checkpoint_params = torch.load(args.pretrained, map_location="cpu")

    # Data Loader
    train_loader, val_loader = get_data_loaders(rank, world_size, args)

    # Model
    model = get_model(world_size, checkpoint_params).eval()

    # Optimizer
    optimizer = torch.optim.SGD(model.classifier.parameters(),
                                lr=args.lr,
                                momentum=0.9,
                                weight_decay=1e-4)

    # Training parameters
    save_params = {"save_dir": os.path.split(args.pretrained)[0]}
    trainer = LinearEvalTrainer(model,
                                optimizer, (train_loader, val_loader),
                                device=device,
                                rank=rank,
                                save_params=save_params)

    # Train
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
