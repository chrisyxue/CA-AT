'''
PGDAT
'''

import os, argparse, time
import numpy as np
# import matplotlib.pyplot as plt
import random
import numpy as np

import torch
import torch.nn.functional as F
# from attacks.cattack import *
import pdb
import copy

import torch.nn.functional as F 

# from mpl_toolkits.mplot3d import Axes3D
from autoattack import AutoAttack

# import torchattacks
import core.attacks.torchattackslib.torchattacks as torchattacks
import pandas as pd
import time
from torch.autograd import Variable

# DDN attacker
from fast_adv.attacks import DDN
from fast_adv.utils.utils import NormalizedModel

def natural(model,
            X,
            y):

    err = (model(X).data.max(1)[1] != y.data).float().sum()
    return err


"""
Untargeted Attack
"""
def pgd20_whitebox(model,
                  X,
                  y,
                  epsilon):
    num_steps=20
    step_size = 2/225
    atk = torchattacks.PGD(model, eps=epsilon, alpha=step_size, steps=num_steps)
    X_adv = atk(X, y)
    err_pgd = (model(X_adv).data.max(1)[1] != y.data).float().sum()
    return err_pgd

def pgd30_whitebox(model,
                  X,
                  y,
                  epsilon):
    num_steps=30
    step_size = 2/225
    atk = torchattacks.PGD(model, eps=epsilon, alpha=step_size, steps=num_steps)
    X_adv = atk(X, y)
    err_pgd = (model(X_adv).data.max(1)[1] != y.data).float().sum()
    return err_pgd

def pgd40_whitebox(model,
                  X,
                  y,
                  epsilon):
    num_steps=40
    step_size = 2/225
    atk = torchattacks.PGD(model, eps=epsilon, alpha=step_size, steps=num_steps)
    X_adv = atk(X, y)
    err_pgd = (model(X_adv).data.max(1)[1] != y.data).float().sum()
    return err_pgd


def pgd_whitebox(model,
                  X,
                  y,
                  epsilon):
    num_steps=10
    step_size = 2/225
    atk = torchattacks.PGD(model, eps=epsilon, alpha=step_size, steps=num_steps)
    X_adv = atk(X, y)
    err_pgd = (model(X_adv).data.max(1)[1] != y.data).float().sum()
    return err_pgd

def pgdl2_whitebox(model,
                  X,
                  y,
                  epsilon):

    num_steps=10
    step_size = 2/225
    atk = torchattacks.PGDL2(model, eps=epsilon, alpha=step_size, steps=num_steps)
    X_adv = atk(X, y)
    err_pgd = (model(X_adv).data.max(1)[1] != y.data).float().sum()
    return err_pgd

def ddn_whitebox(model,
                X,
                y,
                epsilon,steps=100,init_norm=1):
    device = torch.device('cuda')
    
    # normalized
    image_mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
    image_std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)
    model_n = NormalizedModel(model=model, mean=image_mean, std=image_std).to(device)
    
    atk = DDN(steps=steps, device=device, init_norm=init_norm)
    X_adv = atk.attack(model_n, X, labels=y, targeted=False)
    err_pgd = (model_n(X_adv).data.max(1)[1] != y.data).float().sum()
    return err_pgd

def fgsm_whitebox(model,
                X,
                y,
                epsilon):
    atk = torchattacks.FGSM(model, eps=epsilon)
    X_adv = atk(X, y)
    err_pgd = (model(X_adv).data.max(1)[1] != y.data).float().sum()
    return err_pgd

def mifgsm_whitebox(model,
                  X,
                  y,
                  epsilon):
    num_steps=10

    atk = torchattacks.MIFGSM(model, eps=epsilon, steps=num_steps, decay=1.0)
    X_adv = atk(X, y)
    err_pgd = (model(X_adv).data.max(1)[1] != y.data).float().sum()
    return err_pgd

def autopgd_whitebox(model,
                  X,
                  y,
                  epsilon):
    atk = torchattacks.APGD(model, norm='Linf', eps=epsilon)
    X_adv = atk(X, y)
    err_pgd = (model(X_adv).data.max(1)[1] != y.data).float().sum()
    return err_pgd

def autopgddlr_whitebox(model,
                  X,
                  y,
                  epsilon):
    atk = torchattacks.APGD(model, norm='Linf', eps=epsilon, loss='dlr')
    X_adv = atk(X, y)
    err_pgd = (model(X_adv).data.max(1)[1] != y.data).float().sum()
    return err_pgd

def autopgdl2_whitebox(model,
                  X,
                  y,
                  epsilon):
    atk = torchattacks.APGD(model, norm='L2', eps=epsilon)
    X_adv = atk(X, y)
    err_pgd = (model(X_adv).data.max(1)[1] != y.data).float().sum()
    return err_pgd

def fab_whitebox(model,
                  X,
                  y,
                  epsilon):
    num_steps = 10
    atk = torchattacks.FAB(model, eps=epsilon, steps=num_steps, multi_targeted=False)
    X_adv = atk(X, y)
    err_pgd = (model(X_adv).data.max(1)[1] != y.data).float().sum()
    return err_pgd



def eval_untarget_adv_test(model, test_loader, epsilon_linf, epsilon_l2):
    """
    evaluate model by white-box attack
    """
    model.eval()
    natural_err_total = 0
    fgsm_robust_err_total = 0
    pgd_robust_err_total = 0
    pgdl2_robust_err_total = 0
    autopgd_robust_err_total = 0
    autopgdl2_robust_err_total = 0
    fab_robust_err_total = 0
    mifgsm_robust_err_total = 0
    
    count = 1
    for data, target in test_loader:
        print(str(count)+'/'+str(len(test_loader)))
        count = count + 1
        data, target = data.cuda(), target.cuda()
        X, y = Variable(data, requires_grad=True), Variable(target)
        
        # linf-attack
        err_natural = natural(model,X,y)
        pgd_err_robust = pgd_whitebox(model, X, y, epsilon_linf)
        fgsm_err_robust = fgsm_whitebox(model, X, y, epsilon_linf)
        mifgsm_err_robust = mifgsm_whitebox(model, X, y, epsilon_linf)
        autopgd_err_robust = autopgd_whitebox(model, X, y, epsilon_linf)
        fab_err_robust = fab_whitebox(model, X, y, epsilon_linf)
        
        # l2-attack
        pgdl2_err_robust = pgdl2_whitebox(model, X, y, epsilon_l2)
        autopgdl2_err_robust = autopgdl2_whitebox(model, X, y, epsilon_l2)
        fgsm_robust_err_total += fgsm_err_robust
        pgd_robust_err_total += pgd_err_robust
        autopgd_robust_err_total += autopgd_err_robust
        mifgsm_robust_err_total += mifgsm_err_robust
        fab_robust_err_total += fab_err_robust 
        pgdl2_robust_err_total += pgdl2_err_robust
        autopgdl2_robust_err_total += autopgdl2_err_robust

        natural_err_total += err_natural
        # pdb.set_trace()

    natural_acc_total = 1-natural_err_total/len(test_loader.dataset)
    fgsm_robust_acc_total = 1-fgsm_robust_err_total/len(test_loader.dataset)
    pgd_robust_acc_total = 1-pgd_robust_err_total/len(test_loader.dataset)
    autopgd_robust_acc_total = 1-autopgd_robust_err_total/len(test_loader.dataset)
    mifgsm_robust_acc_total = 1-mifgsm_robust_err_total/len(test_loader.dataset)
    fab_robust_acc_total = 1-fab_robust_err_total/len(test_loader.dataset)
    pgdl2_robust_acc_total = 1-pgdl2_robust_err_total/len(test_loader.dataset)
    autopgdl2_robust_acc_total = 1-autopgdl2_robust_err_total/len(test_loader.dataset)

    res = {}
    res['nat'] = natural_acc_total
    res['fgsm'] = fgsm_robust_acc_total 
    res['pgd'] = pgd_robust_acc_total 
    res['autopgd'] = autopgd_robust_acc_total
    res['mifgsm'] = mifgsm_robust_acc_total 
    res['fab'] = fab_robust_acc_total 
    res['pgdl2'] = pgdl2_robust_acc_total 
    res['autopgdl2'] = autopgdl2_robust_acc_total 

    for k in res.keys():
        res[k] = round(res[k].item(),4)
    res['epsilon_linf'] = epsilon_linf
    res['epsilon_l2'] = epsilon_l2
    return res



def eval_untarget_pgd(model, test_loader, epsilon_linf, epsilon_l2):
    """
    evaluate model by white-box attack
    """
    model.eval()
    
    natural_err_total = 0
    pgd_robust_err_total = 0
    pgd20_robust_err_total = 0
    pgd30_robust_err_total = 0
    pgd40_robust_err_total = 0
    
    count = 1
    for data, target in test_loader:
        print(str(count)+'/'+str(len(test_loader)))
        count = count + 1
        data, target = data.cuda(), target.cuda()
        X, y = Variable(data, requires_grad=True), Variable(target)
        
        # linf-attack
        err_natural = natural(model,X,y)
        pgd_err_robust = pgd_whitebox(model, X, y, epsilon_linf)
        pgd20_err_robust = pgd20_whitebox(model, X, y, epsilon_linf)
        pgd30_err_robust = pgd30_whitebox(model, X, y, epsilon_linf)
        pgd40_err_robust = pgd40_whitebox(model, X, y, epsilon_linf)

        pgd_robust_err_total += pgd_err_robust
        pgd20_robust_err_total += pgd20_err_robust
        pgd30_robust_err_total += pgd30_err_robust
        pgd40_robust_err_total += pgd40_err_robust
    
        natural_err_total += err_natural
        # pdb.set_trace()

    natural_acc_total = 1-natural_err_total/len(test_loader.dataset)
    pgd_robust_acc_total = 1-pgd_robust_err_total/len(test_loader.dataset)
    pgd20_robust_acc_total = 1-pgd20_robust_err_total/len(test_loader.dataset)
    pgd30_robust_acc_total = 1-pgd30_robust_err_total/len(test_loader.dataset)
    pgd40_robust_acc_total = 1-pgd40_robust_err_total/len(test_loader.dataset)

    res = {}
    res['nat'] = natural_acc_total
    res['pgd'] = pgd_robust_acc_total 
    res['pgd20'] = pgd20_robust_acc_total 
    res['pgd30'] = pgd30_robust_acc_total 
    res['pgd40'] = pgd40_robust_acc_total 

    for k in res.keys():
        res[k] = round(res[k].item(),4)
    res['epsilon_linf'] = epsilon_linf
    res['epsilon_l2'] = epsilon_l2

    return res