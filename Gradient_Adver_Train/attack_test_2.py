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

from attacks.pdg_grad_proj import *
import torch.nn.functional as F 

from mpl_toolkits.mplot3d import Axes3D
from autoattack import AutoAttack

import torchattacks
import pandas as pd
import time



def _pgd_whitebox(model,
                  X,
                  y,
                  args):
    epsilon=args.eps/255
    num_steps=args.steps
    # step_size=min(epsilon * 1.25, epsilon + 4/255) / num_steps
    step_size = args.step_size

    atk = torchattacks.PGD(model, eps=epsilon, alpha=step_size, steps=num_steps)
    X_adv = atk(X, y)
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_adv = Variable(X_adv.cuda(), requires_grad=True)
    err_pgd = (model(X_adv).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd

def _pgdl2_whitebox(model,
                  X,
                  y,
                  args):
    epsilon=args.eps/255
    num_steps=args.steps
    # step_size=min(epsilon * 1.25, epsilon + 4/255) / num_steps
    step_size = args.step_size

    atk = torchattacks.PGDL2(model, eps=epsilon, alpha=step_size, steps=num_steps)
    X_adv = atk(X, y)
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_adv = Variable(X_adv.cuda(), requires_grad=True)
    err_pgd = (model(X_adv).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd

def _cw_whitebox(model,
                  X,
                  y,
                  args):
    epsilon=args.eps/255
    num_steps=args.steps
    step_size=min(epsilon * 1.25, epsilon + 4/255) / num_steps

    atk = torchattacks.CW(model, c=1, kappa=0, steps=num_steps, lr=0.01)
    X_adv = atk(X, y)
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_adv = Variable(X_adv.cuda(), requires_grad=True)
    err_pgd = (model(X_adv).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd

def _fgsm_whitebox(model,
                  X,
                  y,
                  args):
    epsilon=args.eps/255
    num_steps=args.steps
    # step_size=min(epsilon * 1.25, epsilon + 4/255) / num_steps
    step_size = args.step_size

    atk = torchattacks.FGSM(model, eps=epsilon)
    X_adv = atk(X, y)
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_adv = Variable(X_adv.cuda(), requires_grad=True)
    err_pgd = (model(X_adv).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd

def _mifgsm_whitebox(model,
                  X,
                  y,
                  args):
    epsilon=args.eps/255
    num_steps=args.steps
    # step_size=min(epsilon * 1.25, epsilon + 4/255) / num_steps
    step_size = args.step_size

    atk = torchattacks.MIFGSM(model, eps=epsilon, steps=num_steps, decay=1.0)
    X_adv = atk(X, y)
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_adv = Variable(X_adv.cuda(), requires_grad=True)
    err_pgd = (model(X_adv).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd

def _autopgd_whitebox(model,
                  X,
                  y,
                  args):
    epsilon=args.eps/255
    num_steps=args.steps
    # step_size=min(epsilon * 1.25, epsilon + 4/255) / num_steps
    step_size = args.step_size
    atk = torchattacks.APGD(model, norm='Linf', eps=epsilon, steps=num_steps, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
    X_adv = atk(X, y)
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_adv = Variable(X_adv.cuda(), requires_grad=True)
    err_pgd = (model(X_adv).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd


def _autoattack_whitebox(model,
                  X,
                  y,
                  args):
    epsilon=args.eps/255
    adversary = AutoAttack(model,eps=epsilon,version='standard',norm='Linf')
    X_adv = adversary.run_standard_evaluation(X, y, bs=X.size[0])
    X_adv = Variable(X_adv.cuda(), requires_grad=True)
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    err_pgd = (model(X_adv).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd


def eval_adv_test_whitebox(model, test_loader, args):
    """
    evaluate model by white-box attack
    """
    model.eval()
    fgsm_robust_err_total = 0
    pgd_robust_err_total = 0
    cw_robust_err_total = 0
    pgdl2_robust_err_total = 0
    auto_robust_err_total = 0
    mifgsm_robust_err_total = 0
    natural_err_total = 0

    eps_lst = [i for i in range(8,64,8)]
    res = pd.DataFrame(columns=['PGD','PGDL2','FGSM','CW','Auto','MIFGSM'],index=eps_lst)

    for eps in eps_lst:
        args.eps = eps
        print(args.eps)
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            X, y = Variable(data, requires_grad=True), Variable(target)

            t = time.time()
            err_natural, fgsm_err_robust = _fgsm_whitebox(model, X, y, args)
            err_natural, pgd_err_robust = _pgd_whitebox(model, X, y, args)
            err_natural, cw_err_robust = _cw_whitebox(model, X, y, args)
            err_natural, pgdl2_err_robust = _pgdl2_whitebox(model, X, y, args)
            err_natural, mifgsm_err_robust = _mifgsm_whitebox(model, X, y, args)
            err_natural, auto_err_robust = _autopgd_whitebox(model, X, y, args)
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
            auto_robust_err_total += auto_err_robust
            mifgsm_robust_err_total += mifgsm_err_robust
            pgdl2_robust_err_total += pgdl2_err_robust

            natural_err_total += err_natural
        # pdb.set_trace()

        natural_err_total = 1-natural_err_total/len(test_loader.dataset)
        fgsm_robust_err_total = 1-fgsm_robust_err_total/len(test_loader.dataset)
        pgdl2_robust_err_total = 1-pgdl2_robust_err_total/len(test_loader.dataset)
        pgd_robust_err_total = 1-pgd_robust_err_total/len(test_loader.dataset)
        cw_robust_err_total = 1-cw_robust_err_total/len(test_loader.dataset)
        auto_robust_err_total = 1-auto_robust_err_total/len(test_loader.dataset)
        mifgsm_robust_err_total = 1-mifgsm_robust_err_total/len(test_loader.dataset)


        res.loc[eps,:] = [pgd_robust_err_total.item(),pgdl2_robust_err_total.item(),fgsm_robust_err_total.item(),cw_robust_err_total.item(),auto_robust_err_total.item(),mifgsm_robust_err_total.item()]

        print('natural_acc_total: ', 1-natural_err_total/len(test_loader.dataset))
        print('fgsm_acc_total: ', 1-fgsm_robust_err_total/len(test_loader.dataset))
        print('pgd_acc_total: ', 1-pgd_robust_err_total/len(test_loader.dataset))
        print('cw_acc_total: ', 1-cw_robust_err_total/len(test_loader.dataset))

    return res


def main():
    model = nn.DataParallel(getattr(models, 'WideResNet')(num_classes=10)).cuda()
    model.load_state_dict(torch.load(path))
    eval_adv_test_whitebox(model, device, test_loader, args.attack_method)


if __name__ == '__main__':
    main()