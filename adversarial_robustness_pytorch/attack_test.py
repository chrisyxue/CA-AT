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

def natural(model,
            X,
            y):

    err = (model(X).data.max(1)[1] != y.data).float().sum()
    return err


"""
Untargeted Attack
"""
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

def cw_whitebox(model,
                X,
                y,
                epsilon):
    
    num_steps=50
    step_size = 2/225
    atk = torchattacks.CW(model, c=1, kappa=0, steps=num_steps, lr=0.01)
    X_adv = atk(X, y)
    err_pgd = (model(X_adv).data.max(1)[1] != y.data).float().sum()
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


"""
Targeted Attack
"""

def tfab_whitebox(model,
                  X,
                  y,
                  epsilon,
                  n_classes):
    num_steps = 10
    atk = torchattacks.FAB(model, eps=epsilon, steps=num_steps, multi_targeted=True, n_classes=n_classes)
    X_adv = atk(X, y)
    err_pgd = (model(X_adv).data.max(1)[1] != y.data).float().sum()
    return err_pgd

def tapgd_whitebox(model,
                  X,
                  y,
                  epsilon,
                  n_classes):
    atk =  torchattacks.APGDT(model, norm='Linf', eps=epsilon, n_classes=n_classes)
    X_adv = atk(X, y)
    err_pgd = (model(X_adv).data.max(1)[1] != y.data).float().sum()
    return err_pgd

def tapgdl2_whitebox(model,
                  X,
                  y,
                  epsilon,
                  n_classes):
    atk =  torchattacks.APGDT(model, norm='L2', eps=epsilon, n_classes=n_classes)
    X_adv = atk(X, y)
    err_pgd = (model(X_adv).data.max(1)[1] != y.data).float().sum()
    return err_pgd


def eval_target_adv_test(model, test_loader, epsilon_linf, epsilon_l2,n_classes):
    """
    evaluate model by white-box attack
    """
    model.eval()
    natural_err_total = 0

    tautopgd_robust_err_total = 0
    tautopgdl2_robust_err_total = 0
    tfab_robust_err_total = 0
    count = 1
    for data, target in test_loader:
        print(str(count)+'/'+str(len(test_loader)))
        count = count + 1
        data, target = data.cuda(), target.cuda()
        X, y = Variable(data, requires_grad=True), Variable(target)
        
        # linf-attack
        n_classes = 10
        err_natural = natural(model,X,y)

        tautopgd_err_robust = tapgd_whitebox(model, X, y, epsilon_linf, n_classes)
        tautopgdl2_err_robust = tapgdl2_whitebox(model, X, y, epsilon_l2, n_classes)
        tfab_err_robust = tfab_whitebox(model, X, y, epsilon_linf, n_classes)

        natural_err_total += err_natural
        tautopgd_robust_err_total += tautopgd_err_robust
        tautopgdl2_robust_err_total += tautopgdl2_err_robust
        tfab_robust_err_total += tfab_err_robust
    
    natural_acc_total = 1-natural_err_total/len(test_loader.dataset)
    tautopgd_acc_robust = 1-tautopgd_robust_err_total/len(test_loader.dataset)
    tautopgdl2_acc_robust = 1-tautopgdl2_robust_err_total/len(test_loader.dataset)
    tfab_acc_robust = 1-tfab_robust_err_total/len(test_loader.dataset)

    res = {}
    res['nat'] = natural_acc_total
    res['t-apgd'] = tautopgd_acc_robust 
    res['t-apgdl2'] = tautopgdl2_acc_robust
    res['t-fab'] = tfab_acc_robust

    for k in res.keys():
        res[k] = round(res[k].item(),4)
    return res



def eval_untarget_adv_test(model, test_loader, epsilon_linf, epsilon_l2):
    """
    evaluate model by white-box attack
    """
    model.eval()
    natural_err_total = 0
    fgsm_robust_err_total = 0
    pgd_robust_err_total = 0
    cw_robust_err_total = 0
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
        cw_err_robust = cw_whitebox(model, X, y, epsilon_linf)
        mifgsm_err_robust = mifgsm_whitebox(model, X, y, epsilon_linf)
        autopgd_err_robust = autopgd_whitebox(model, X, y, epsilon_linf)
        fab_err_robust = fab_whitebox(model, X, y, epsilon_linf)
        
        # l2-attack
        pgdl2_err_robust = pgdl2_whitebox(model, X, y, epsilon_l2)
        autopgdl2_err_robust = autopgdl2_whitebox(model, X, y, epsilon_l2)
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
    cw_robust_acc_total = 1-cw_robust_err_total/len(test_loader.dataset)
    autopgd_robust_acc_total = 1-autopgd_robust_err_total/len(test_loader.dataset)
    mifgsm_robust_acc_total = 1-mifgsm_robust_err_total/len(test_loader.dataset)
    fab_robust_acc_total = 1-fab_robust_err_total/len(test_loader.dataset)
    pgdl2_robust_acc_total = 1-pgdl2_robust_err_total/len(test_loader.dataset)
    autopgdl2_robust_acc_total = 1-autopgdl2_robust_err_total/len(test_loader.dataset)

    res = {}
    res['nat'] = natural_acc_total
    res['fgsm'] = fgsm_robust_acc_total 
    res['pgd'] = pgd_robust_acc_total 
    res['cw'] = cw_robust_acc_total
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

