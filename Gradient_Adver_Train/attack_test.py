'''
PGDAT
'''

import os, argparse, time
import numpy as np
import matplotlib.pyplot as plt
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam, lr_scheduler

from models.cifar10.resnet import ResNet34
from models.svhn.wide_resnet import WRN16_8
from models.stl10.wide_resnet import WRN40_2

from dataloaders.cifar10 import cifar10_dataloaders
from dataloaders.svhn import svhn_dataloaders
from dataloaders.stl10 import stl10_dataloaders

from utils.utils import *
from utils.context import ctx_noparamgrad_and_eval
from attacks.pgd import PGD
# from attacks.cattack import *
import pdb
import copy
from torchdiffeq import odeint
from attacks.pdg_grad_proj import *
import torch.nn.functional as F 

from mpl_toolkits.mplot3d import Axes3D

import torchattacks

def linf_clamp(x, _min, _max):
    '''
    Inplace linf clamping on Tensor x.

    Args:
        x: Tensor. shape=(N,C,W,H)
        _min: Tensor with same shape as x.
        _max: Tensor with same shape as x.
    '''
    idx = x.data < _min
    x.data[idx] = _min[idx]
    idx = x.data > _max
    x.data[idx] = _max[idx]

    return x
    
def one_hot_tensor(y_batch_tensor, num_classes, device):
    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0),
                                      num_classes).fill_(0)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor

class CWLoss(nn.Module):
    def __init__(self, num_classes, margin=50, reduce=True):
        super(CWLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.reduce = reduce
        return

    def forward(self, logits, targets):
        """
        :param inputs: predictions
        :param targets: target labels
        :return: loss
        """
        onehot_targets = one_hot_tensor(targets, self.num_classes,
                                        targets.device)

        self_loss = torch.sum(onehot_targets * logits, dim=1)
        other_loss = torch.max(
            (1 - onehot_targets) * logits - onehot_targets * 1000, dim=1)[0]

        loss = -torch.sum(torch.clamp(self_loss - other_loss + self.margin, 0))

        if self.reduce:
            sample_num = onehot_targets.shape[0]
            loss = loss / sample_num

        return loss

def _pgd_whitebox(model,
                  X,
                  y,
                  args):
    epsilon=args.eps/255
    num_steps=args.steps
    step_size=min(epsilon * 1.25, epsilon + 4/255) / num_steps

    atk = torchattacks.PGD(model, eps=epsilon, alpha=step_size, steps=num_steps)
    X_adv = atk(X, y)
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    # X_adv = X.clone()
    # X_adv += (2.0 * torch.rand(X_adv.shape).cuda() - 1.0) * epsilon # random initialize
    # X_adv = torch.clamp(X_adv, 0, 1) # clamp to RGB range [0,1]
    X_adv = Variable(X_adv.cuda(), requires_grad=True)
    
    # for _ in range(num_steps):
    #     model.zero_grad()

    #     with torch.enable_grad():
    #         loss = nn.CrossEntropyLoss(reduction="sum")(model(X_adv), y)
    #     loss.backward()
    #     eta = step_size * X_adv.grad.data.sign()
    #     X_adv = Variable(X_adv.data + eta, requires_grad=True)
    #     eta = torch.clamp(X_adv.data - X.data, -epsilon, epsilon)
    #     X_adv = Variable(X.data + eta, requires_grad=True)
    #     X_adv = Variable(torch.clamp(X_adv, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_adv).data.max(1)[1] != y.data).float().sum()
    #print('err pgd (white-box): ', err_pgd)
    return err, err_pgd


def _cw_whitebox(model,
                  X,
                  y,
                  args):
    epsilon=args.eps/255
    num_steps=args.steps
    step_size=min(epsilon * 1.25, epsilon + 4/255) / num_steps

    atk = torchattacks.PGD(model, eps=epsilon, alpha=step_size, steps=num_steps)
    X_adv = atk(X, y)
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    # X_adv = X.clone()
    # X_adv += (2.0 * torch.rand(X_adv.shape).cuda() - 1.0) * epsilon # random initialize
    # X_adv = torch.clamp(X_adv, 0, 1) # clamp to RGB range [0,1]
    X_adv = Variable(X_adv.cuda(), requires_grad=True)
    

    for _ in range(num_steps):
        model.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss(reduction="sum")(model(X_adv), y)
        loss.backward()
        eta = step_size * X_adv.grad.data.sign()
        X_adv = Variable(X_adv.data + eta, requires_grad=True)
        eta = torch.clamp(X_adv.data - X.data, -epsilon, epsilon)
        X_adv = Variable(X.data + eta, requires_grad=True)
        X_adv = Variable(torch.clamp(X_adv, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_adv).data.max(1)[1] != y.data).float().sum()
    #print('err pgd (white-box): ', err_pgd)
    return err, err_pgd


def _cw_whitebox(model,
                 X,
                 y,
                 args):
    epsilon=args.eps/255
    num_steps=args.steps
    # step_size=args.step_size
    step_size=min(epsilon * 1.25, epsilon + 4/255) / num_steps
    
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = X.clone()
    X_pgd += (2.0 * torch.rand(X_pgd.shape).cuda() - 1.0) * epsilon # random initialize
    X_pgd = torch.clamp(X_pgd, 0, 1) # clamp to RGB range [0,1]
    X_pgd = Variable(X_pgd.cuda(), requires_grad=True)

    for _ in range(num_steps):
        model.zero_grad()

        with torch.enable_grad():
            loss = CWLoss(100 if args.dataset == 'cifar100' else 10)(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    #print('err cw (white-box): ', err_pgd)
    return err, err_pgd


def _fgsm_whitebox(model,
                  X,
                  y,
                  args):
    epsilon=args.eps/255
    num_steps = 1
    step_size=args.eps/255

    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)

    for _ in range(num_steps):
        model.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    #print('err pgd (white-box): ', err_pgd)
    return err, err_pgd


def _mim_whitebox(model,
                  X,
                  y,
                  args,
                  decay_factor=1.0):
    epsilon=args.eps/255
    num_steps = 1
    # step_size=args.eps/255
    step_size=min(epsilon * 1.25, epsilon + 4/255) / num_steps

    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()

    X_pgd = X.clone()
    X_pgd += (2.0 * torch.rand(X_pgd.shape).cuda() - 1.0) * epsilon # random initialize
    X_pgd = torch.clamp(X_pgd, 0, 1) # clamp to RGB range [0,1]
    X_pgd = Variable(X_pgd.cuda(), requires_grad=True)
    
    previous_grad = torch.zeros_like(X.data)

    for _ in range(num_steps):
        model.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        grad = X_pgd.grad.data / torch.mean(torch.abs(X_pgd.grad.data), [1,2,3], keepdim=True)
        previous_grad = decay_factor * previous_grad + grad
        X_pgd = Variable(X_pgd.data + step_size * previous_grad.sign(), requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    #print('err mim (white-box): ', err_pgd)
    return err, err_pgd

def eval_adv_test_whitebox(model, test_loader, args):
    """
    evaluate model by white-box attack
    """
    model.eval()
    fgsm_robust_err_total = 0
    pgd_robust_err_total = 0
    cw_robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, fgsm_err_robust = _fgsm_whitebox(model, X, y, args)
        err_natural, pgd_err_robust = _pgd_whitebox(model, X, y, args)
        err_natural, cw_err_robust = _cw_whitebox(model, X, y, args)
        '''
        if attack_method == 'PGD':
            err_natural, err_robust = _pgd_whitebox(model, X, y)
        elif attack_method == 'CW':
            err_natural, err_robust = _cw_whitebox(model, X, y)
        elif attack_method == 'MIM':
            err_natural, err_robust = _mim_whitebox(model, X, y)
        elif attack_method == 'FGSM':
            err_natural, err_robust = _fgsm_whitebox(model, X, y)
        '''
        fgsm_robust_err_total += fgsm_err_robust
        pgd_robust_err_total += pgd_err_robust
        cw_robust_err_total += cw_err_robust
        natural_err_total += err_natural
    print('natural_acc_total: ', 1-natural_err_total/len(test_loader.dataset))
    print('fgsm_acc_total: ', 1-fgsm_robust_err_total/len(test_loader.dataset))
    print('pgd_acc_total: ', 1-pgd_robust_err_total/len(test_loader.dataset))
    print('cw_acc_total: ', 1-cw_robust_err_total/len(test_loader.dataset))


def main():
    model = nn.DataParallel(getattr(models, 'WideResNet')(num_classes=10)).cuda()
    model.load_state_dict(torch.load(path))
    eval_adv_test_whitebox(model, device, test_loader, args.attack_method)


if __name__ == '__main__':
    main()