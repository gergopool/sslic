import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, datasets
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
import os
from typing import Union
import pkbar

from .utils import AllReduce, AllGather


def get_loaders(train_dataset, val_dataset, batch_size=256):

    world_size, rank = get_world_size_n_rank()

    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(train_dataset,
                              num_workers=16,
                              pin_memory=True,
                              persistent_workers=True,
                              batch_size=batch_size,
                              sampler=train_sampler)
    val_loader = DataLoader(val_dataset,
                            num_workers=16,
                            pin_memory=True,
                            persistent_workers=True,
                            batch_size=batch_size,
                            sampler=val_sampler)

    return train_loader, val_loader


def cifar10(root, batch_size=256):

    trans = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    train_dataset = datasets.CIFAR10(root, transform=trans, train=True)
    val_dataset = datasets.CIFAR10(root, transform=trans, train=False)

    return get_loaders(train_dataset, val_dataset, batch_size)


def cifar100(root, batch_size=256):

    trans = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
    ])

    train_dataset = datasets.CIFAR100(root, transform=trans, train=True)
    val_dataset = datasets.CIFAR100(root, transform=trans, train=False)

    return get_loaders(train_dataset, val_dataset, batch_size)


def imagenet(root, batch_size=256):

    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_root = os.path.join(root, 'train')
    val_root = os.path.join(root, 'val')
    train_dataset = datasets.ImageFolder(train_root, trans)
    val_dataset = datasets.ImageFolder(val_root, trans)

    return get_loaders(train_dataset, val_dataset, batch_size)


def get_loaders_by_name(root, dataset_name, **kwargs):
    if dataset_name in ['imagenet', 'cifar10', 'cifar100']:
        return globals()[dataset_name](root, **kwargs)
    else:
        raise NameError(
            f"Unknown dataset name: {dataset_name}. Please choose from [imagenet, cifar10, cifar100]"
        )


def get_world_size_n_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    else:
        return 1, 0


def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """_accuracy 
    Accuracy of the model
    """
    pred = torch.max(y_hat.data, 1)[1]
    acc = (pred == y).sum() / len(y)
    return AllReduce.apply(acc)


class KNNEvaluator:

    def __init__(self,
                 model: nn.Module,
                 cnn_dim: int,
                 dataset: str,
                 root: str,
                 batch_size=256,
                 verbose=True):
        self.model = model
        self.cnn_dim = cnn_dim
        self.dataset = dataset
        self.batch_size = batch_size
        self.verbose = verbose and get_world_size_n_rank()[1] == 0

        data_loaders = get_loaders_by_name(root, dataset, batch_size=batch_size)
        self.train_loader, self.val_loader = data_loaders

    def _get_labeled_embeddings(self):

        features, labels = [], []

        if self.verbose:
            pbar = pkbar.Pbar(name='Generating embeddings', target=len(self.train_loader))

        for i, (x, y) in enumerate(self.train_loader):
            x = x.cuda()
            y = y.cuda()
            z = self.model(x)

            features.append(z)
            labels.append(y)

            if self.verbose:
                pbar.update(i)

        features = AllGather.apply(torch.cat(features))
        labels = AllGather.apply(torch.cat(labels))

        return features, labels

    def _lin_eval(self,
                  train_z: torch.Tensor,
                  train_y: torch.Tensor,
                  epochs: int = 100) -> torch.Tensor:
        classifier = nn.Linear(self.cnn_dim, train_y.max() + 1).cuda()
        classifier.weight.data.normal_(mean=0.0, std=0.01)
        classifier.bias.data.zero_()

        if get_world_size_n_rank()[0] > 1:
            classifier = nn.parallel.DistributedDataParallel(classifier)

        opt = torch.optim.SGD(classifier.parameters(), lr=1e-2, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        batch_size = 16384 if self.dataset == 'imagenet' else 512
        ipe = len(train_z) // batch_size

        if self.verbose:
            pbar = pkbar.Pbar(f"Linear Eval - Training {epochs} epochs.", target=epochs * ipe)

        for epoch in range(epochs):
            indices = torch.randperm(len(train_z))
            for i in range(ipe):
                start = i * batch_size
                end = start + batch_size
                j = indices[start:end]
                x = train_z[j].float()
                y = train_y[j]

                y_hat = classifier(x)
                loss = criterion(y_hat, y)
                opt.zero_grad()
                loss.backward()
                opt.step()

                if self.verbose:
                    pbar.update(epoch * ipe + i)

        if self.verbose:
            print(f"Linear Eval - Evaluating.")
            pbar = pkbar.Kbar(target=len(self.val_loader))

        accs = []
        for x, y in self.val_loader:
            x = x.cuda()
            y = y.cuda()
            with torch.no_grad():
                y_hat = classifier(self.model(x))
            acc = accuracy(y_hat, y)
            accs.append(acc)
            if self.verbose:
                pbar.add(1, values=[("acc", acc)])

        acc = AllReduce.apply(torch.Tensor(accs).cuda()).mean()

        if self.verbose:
            print(f"Top1 @ Linear Eval: {acc*100:3.2f}%")

        return acc

    def _evaluate(self, train_z: torch.Tensor, train_y: torch.Tensor, ks: list) -> torch.Tensor:

        train_z = F.normalize(train_z, dim=1)
        ks = [int(k) for k in ks]
        largest_k = max(ks)
        results = torch.zeros(len(ks)).cuda()
        train_y = train_y.repeat(self.batch_size).view(self.batch_size, -1)

        if self.verbose:
            print("KNN-evaluation")
            pbar = pkbar.Kbar(target=len(self.val_loader))

        total = 0
        for x, y in self.val_loader:
            x = x.cuda()
            y = y.cuda()
            z = F.normalize(self.model(x))

            batch_results = torch.zeros(len(ks)).cuda()

            dist = z @ train_z.T
            closest_indices = dist.topk(largest_k, dim=1)[1]
            pred_labels = torch.gather(train_y, dim=1, index=closest_indices)
            for j, k in enumerate(ks):
                preds = pred_labels[:, :k].mode(dim=1)[0]
                hits = (preds == y).sum()
                batch_results[j] += hits

            results += batch_results
            total += len(y)

            if self.verbose:
                accs = list(batch_results.cpu().numpy() * 100 / len(y))
                update_values = [(f"k={k}", acc) for (k, acc) in zip(ks, accs)]
                pbar.add(1, values=update_values)

        results /= total
        results = AllReduce.apply(results)

        if self.verbose:
            accs = list(results.cpu().numpy())
            for k, acc in zip(ks, accs):
                print(f"Top1 @ K={k:<2d} : {acc*100:3.2f}%")

        return results

    def __call__(self, ks: Union[int, list] = 1) -> torch.Tensor:

        if isinstance(ks, int):
            ks = [ks]

        if not isinstance(ks, list):
            raise TypeError(f"Value k must be either a list or int, not {type(ks)}")

        # Freee batchnorm
        training_mode_used_before = self.model.training
        self.model.eval()

        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                train_features, train_labels = self._get_labeled_embeddings()
                results = self._evaluate(train_features, train_labels, ks)

        self._lin_eval(train_features, train_labels)

        self.model.train(training_mode_used_before)

        return results
