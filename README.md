# Self-supervised Image Classification in PyTorch

 - [x] Training
 - [x] Linear eval
 - [x] Soft-nearest neighbor online classifier
 - [x] Warmup cosine schedule
 - [x] Lars
 - [x] Multi-GPU training
 - [ ] Tensorboard
 - [ ] Reproduction of results published

## Implemented algorithms

 * SimCLR [simclr]
 * SimSiam [simsiam]
 * Barlow Twins [barlow_twins]

## Implemented datasets

 * CIFAR10 [cifar10]
 * CIFAR100 [cifar100]
 * ImageNet [imagenet]

## Train

```
python train.py simsiam path/to/cifar10/root --devices 0
```

You can replace simsiam wih any algorithms described above. See parsed arguments in code for other options.

## Linear Eval

```
python lin_eval.py path/to/checkpoint.pth.tar path/to/cifar10/root --devices 0 1
```

## Credits

Some parts of the code was inherited from different repositories of the facebookresearch team, so huge credit goes there.