# Copied from: https://github.com/lvyilin/pytorch-fgvc-dataset/blob/master/tiny_imagenet.py

import os
import os
import pandas as pd
import warnings
from torchvision.datasets import ImageNet
import torchvision.transforms as transforms

DATA_DESC = {
    'data': 'imagenet',
    'classes': tuple(range(0, 1000)),
    'num_classes': 1000,
    'mean': [0.4802, 0.4481, 0.3975], 
    'std': [0.2302, 0.2265, 0.2262],
}


def load_imagenet(args, data_dir, use_augmentation=False):
    """
    Returns ImageNet-200 train, test datasets and dataloaders.
    Arguments:
        data_dir (str): path to data directory.
        use_augmentation (bool): whether to use augmentations for training set.
    Returns:
        train dataset, test dataset. 
    """
    test_transform = transforms.Compose([transforms.ToTensor()])
    if use_augmentation:
        train_transform = transforms.Compose([transforms.RandomCrop(64, padding=4), transforms.RandomHorizontalFlip(), 
                                              transforms.ToTensor()])
    else: 
        train_transform = test_transform

    train_dataset = ImageNet(data_dir, split='train', download=True, transform=train_transform)
    test_dataset = ImageNet(data_dir, split='val', download=False, transform=test_transform)
    # train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    # # train_dataset, test_dataset = random_split(train_dataset, lengths=[80000, 20000])
    # test_dataset = ImageFolder(os.path.join(data_dir, 'val_new'), transform=test_transform)
    # assert(test_dataset.class_to_idx == train_dataset.class_to_idx)
    # import pdb; pdb.set_trace()
    return train_dataset, test_dataset



