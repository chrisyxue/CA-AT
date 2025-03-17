import os
import argparse
import datetime
import numpy as np
import _pickle as pickle
import random
import torch


PRETRAIN_PATH_CIFAR10 = {
    'resnet18': '/scratch/zx1673/results/xuezhiyu/adver_robustness_notrack/cifar10_aug_wa0.995/1_resnet18_adver_avg_beta_0.0/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4/weights-last.pt',
    'resnet34': '/scratch/zx1673/results/xuezhiyu/adver_robustness_notrack/cifar10_aug_wa0.995/1_resnet34_adver_avg_beta_0.0/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4/weights-last.pt'
}
PRETRAIN_PATH_CIFAR100 = {
    'resnet18': '/scratch/zx1673/results/xuezhiyu/adver_robustness_notrack/cifar100_aug_wa0.995/1_resnet18_adver_avg_beta_0.0/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4/weights-last.pt',
    'resnet34': '/scratch/zx1673/results/xuezhiyu/adver_robustness_notrack/cifar100_aug_wa0.995/1_resnet34_adver_avg_beta_0.0/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4/weights-last.pt'
}

PRETRAIN_PATH_TINY = {
    'preact-resnet18':'/scratch/zx1673/results/xuezhiyu/adver_robustness_notrack/tiny-imagenet_aug/1_preact-resnet18_adver_avg_beta_0.0/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4/weights-last.pt',
    'preact-resnet34':'/scratch/zx1673/results/xuezhiyu/adver_robustness_notrack/tiny-imagenet_aug/1_preact-resnet34_adver_avg_beta_0.0/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4/weights-last.pt'
}

PREREAIN_PATH_IMAGENET = {
    'resnet18':'/scratch/zx1673/dataset/models/torch/resnet18-5c106cde.pth',
    'resnet34':'/scratch/zx1673/dataset/models/torch/resnet34-333f7ec4.pth'
}


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def get_pretrained(args,model):
    if args.pretrain_type == 'No':
        return model
    elif args.pretrain_type == 'In-Domain':
        if args.data == 'cifar10':
            path = PRETRAIN_PATH_CIFAR10[args.model]
        elif args.data == 'cifar100':
            path = PRETRAIN_PATH_CIFAR100[args.model]
        elif args.data == 'tiny-imagenet':
            path = PRETRAIN_PATH_TINY[args.model]
        else:
            raise ValueError('Dataset name is not valid')
        param_dict = torch.load(path)['model_state_dict']
        del param_dict['module.0.linear.weight']
        del param_dict['module.0.linear.bias']
        model.load_state_dict(param_dict,strict=False)
        print('Load Pretrained Model on ' + path)
        return model
    elif args.pretrain_type == 'ImageNet':
        path = PREREAIN_PATH_IMAGENET[args.model]
        param_dict = torch.load(path)
        del param_dict['fc.weight']
        del param_dict['fc.bias']
        model.load_state_dict(param_dict)
        return model
    else:
        raise ValueError('Pretrain is not valid')


class SmoothCrossEntropyLoss(torch.nn.Module):
    """
    Cross entropy loss with label smoothing.
    """
    def __init__(self, smoothing=0.0, reduction='mean'):
        super(SmoothCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.reduction = reduction

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def track_bn_stats(model, track_stats=True):
    """
    If track_stats=False, do not update BN running mean and variance and vice versa.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = track_stats


def set_bn_momentum(model, momentum=1):
    """
    Set the value of momentum for all BN layers.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = momentum


def str2bool(v):
    """
    Parse boolean using argument parser.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2float(x):
    """
    Parse float and fractions using argument parser.
    """
    if '/' in x:
        n, d = x.split('/')
        return float(n)/float(d)
    else:
        try:
            return float(x)
        except:
            raise argparse.ArgumentTypeError('Fraction or float value expected.')


def format_time(elapsed):
    """
    Format time for displaying.
    Arguments:
        elapsed: time interval in seconds.
    """
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def seed(seed=1):
    """
    Seed for PyTorch reproducibility.
    Arguments:
        seed (int): Random seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def unpickle_data(filename, mode='rb'):
    """
    Read data from pickled file.
    Arguments:
        filename (str): path to the pickled file.
        mode (str): read mode.
    """
    with open(filename, mode) as pkfile:
        data = pickle.load(pkfile)
    return data


def pickle_data(data, filename, mode='wb'):
    """
    Write data to pickled file.
    Arguments:
        data (Any): data to be written.
        filename (str): path to the pickled file.
        mode (str): write mode.
    """
    with open(filename, mode) as pkfile:
         pickle.dump(data, pkfile)


class NumpyToTensor(object):
    """
    Transforms a numpy.ndarray to torch.Tensor.
    """
    def __call__(self, sample): 
        return torch.from_numpy(sample)


def render_resfile_name(args):
    seed = args.seed
    model = args.model
    adver_loss = args.adver_loss
    grad_op = args.grad_op
    data = args.data
    batch_size = args.batch_size
    epochs = args.num_adv_epochs
    beta = args.beta
    grad_thres = args.grad_thres
    grad_norm = args.grad_norm
    aug = args.augment
    lr = args.lr
    tau = args.tau
    attack = args.attack
    eps = args.attack_eps
    step_size = args.attack_step
    iter = args.attack_iter
    classifier = args.classifier
    adaptive_thres = args.adaptive_thres
    pretrain_type = args.pretrain_type
    adver_criterion = args.adver_criterion
    grad_track_multiattack = args.grad_track_multiattack

    use_ls = args.use_ls
    ls_factor = args.ls_factor

    dir1 = str(data)
    if args.use_pretrain:
        dir1 += '_pretrain'
        
    if aug == True:
        dir1 += '_aug'
    if tau > 0:
        dir1 += '_wa'+str(tau)

    if pretrain_type != 'No':
        if pretrain_type == 'In-Domain':
            dir1 += '_pretrain_InDomain'
        else:
            dir1 += '_pretrain_ImageNet'
    
    dir1 += '_adver_criterion_' + str(adver_criterion)
    if grad_track_multiattack:
        dir1 += '_grad_track_multiattack'
    
    dir2 = str(seed)+'_'+str(model)+'_'+str(adver_loss)+'_'+str(grad_op)

    if adaptive_thres == 'No':
        if grad_op == 'avg':
            dir2 += '_beta_'+str(beta)
        elif 'thres' in grad_op:
            dir2 += '_thres_'+str(grad_thres)
    else:
        if grad_op == 'avg':
            dir2 += '_beta_'+str(beta)
        elif 'thres' in grad_op:
            dir2 += '_thres_ad_'+str(adaptive_thres)
    
    if grad_norm:
        dir2 += '_gradnorm'
    if classifier != 'linear':
        dir2 += '_cls_' + classifier
    
    if use_ls:
        dir2 += '_ls_' + str(ls_factor)
    
    dir3 = str(attack)+'_'+str(eps)+'_'+str(step_size)+'_'+str(iter)
    dir4 = str(epochs)+'_'+str(batch_size)+'_'+str(lr)

    dir = os.path.join(args.log_dir,dir1,dir2,dir3,dir4)
    if os.path.exists(dir) == False:
        os.makedirs(dir)
    return dir


    
