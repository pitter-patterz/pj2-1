# Introduction
This is the first task of *project 2*. We use Resnet-18 to do classfication on CIFAR images, and implemenet three methods of data augmentation, including cutout, mixup and cutmix. The model is referred to https://github.com/samcw/ResNet18-Pytorch.

+ *model.py*: Define the structure of Resnet-18.
+ *data_augmentation.py*: Mixup and cutmix. Note that cutout is realised via transforms.RandomCrop().
+ *train.py**: Train a Resnet-18 within 10 epochs.
+ *test.py*: Test the accuracy of our trained net, while inspecting some misclassified images.
+ *record/*: Loss and accuracy curves.
+ *net.pth*: The trained net.

# Usage
For each of the four training modes (none, cutout, mixup and cutmix), we try three different learning rates 0.0007, 0.001 and 0.0012. Use commands to set the trainig mode.

> python train.py  (same as python train.py none)
> python train.py cutout
> python train.py cutmix
> python train.py mixup


