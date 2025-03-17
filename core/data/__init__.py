import os
import torch

from .cifar10 import load_cifar10
from .cifar100 import load_cifar100
from .svhn import load_svhn
from .tiny_imagenet import load_tinyimagenet
from .imagenet import load_imagenet

DATASETS = ['cifar10', 'svhn', 'cifar100', 'tiny-imagenet']

_LOAD_DATASET_FN = {
    'cifar10': load_cifar10,
    'cifar100': load_cifar100,
    'svhn': load_svhn,
    'tiny-imagenet': load_tinyimagenet,
    'imagenet': load_imagenet
}


def get_data_info(data_dir):
    """
    Returns dataset information.
    Arguments:
        data_dir (str): path to data directory.
    """
    dataset = os.path.basename(os.path.normpath(data_dir))
    if 'cifar100' in data_dir:
        from .cifar100 import DATA_DESC
    elif 'cifar10' in data_dir:
        from .cifar10 import DATA_DESC
    elif 'svhn' in data_dir:
        from .svhn import DATA_DESC
    elif 'tiny-imagenet' in data_dir:
        from .tiny_imagenet import DATA_DESC
    elif 'imagenet' in data_dir:
        from .imagenet import DATA_DESC
    else:
        raise ValueError(f'Only data in {DATASETS} are supported!')
    DATA_DESC['data'] = dataset
    return DATA_DESC


def load_data(args, data_dir, batch_size=256, batch_size_test=256, num_workers=1, use_augmentation=False, shuffle_train=True):
    """
    Returns train, test datasets and dataloaders.
    Arguments:
        data_dir (str): path to data directory.
        batch_size (int): batch size for training.
        batch_size_test (int): batch size for validation.
        num_workers (int): number of workers for loading the data.
        use_augmentation (bool): whether to use augmentations for training set.
        shuffle_train (bool): whether to shuffle training set.
        aux_data_filename (str): path to unlabelled data.
        unsup_fraction (float): fraction of unlabelled data per batch.
    """
    dataset = os.path.basename(os.path.normpath(data_dir))
    load_dataset_fn = _LOAD_DATASET_FN[dataset]
    
   
    train_dataset, test_dataset = load_dataset_fn(args=args, data_dir=data_dir, use_augmentation=use_augmentation)
    pin_memory = torch.cuda.is_available()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, 
                                                    num_workers=num_workers, pin_memory=pin_memory)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, 
                                                    num_workers=num_workers, pin_memory=pin_memory)
    
    return train_dataset, test_dataset, train_dataloader, test_dataloader
