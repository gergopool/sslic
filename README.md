# Self-supervised Image Classification in PyTorch

 - [x] Training
 - [x] Linear eval
 - [x] Soft-nearest neighbor online classifier
 - [x] Warmup cosine schedule
 - [x] Lars
 - [x] Multi-GPU training
 - [x] Tensorboard
 - [ ] Reproduction of results published

## Implemented algorithms

 * SimCLR
 * SimSiam
 * Barlow Twins
 * BYOL
 * MocoV2
 * ReSSL
 * TWIST
 * VICReg

## Implemented datasets

 * CIFAR10
 * CIFAR100
 * ImageNet
 * Tiny-ImageNet

## Train

```
python train.py ressl path/to/cifar10/root --dataset cifar10 --devices 0
```

You can replace simsiam wih any algorithms described above. See parsed arguments in code for other options.

## Linear Eval

*Currently not working and only avaiable during training. The evaluation results shown during training are very good estimates, although not official results and might be below the published accuracies by 1-2%. They are only made for quick estimates on the performance of the encoder network.*

## Credits

Some parts of the code was inherited from different repositories of the facebookresearch team, so huge credit goes there. Also special thanks to:
 - [@koszpe](https://github.com/koszpe) for adding tensorboard.
