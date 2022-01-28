import argparse
import torch

from sslic.trainer import SSLTrainer
from sslic.models import get_ssl_method
from sslic.data import get_dataset_provider

parser = argparse.ArgumentParser(description='Simple settings.')
parser.add_argument('method', type=str, choices=['simsiam', 'simclr', 'barlow_twins'])
parser.add_argument('data_root', type=str)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-1)


def main():
    args = parser.parse_args()

    # Data Loader
    dataset_provider = get_dataset_provider(args.data_root, args.method, args.dataset)
    train_loader = torch.utils.data.DataLoader(dataset_provider('ssl'),
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=8,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset_provider('test'),
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=8,
                                             pin_memory=True)

    # Model
    model = get_ssl_method(args.method, args.dataset, n_classes=dataset_provider.n_classes)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = SSLTrainer(model, optimizer, (train_loader, val_loader), device=device)
    trainer.train(args.epochs)


if __name__ == '__main__':
    main()