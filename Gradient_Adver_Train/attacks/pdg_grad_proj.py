import torch
import torch.nn.functional as F
import torch
from torch.utils.data import random_split
import copy
from attacks.trades import trades_loss
import pdb
import math
import sys

from attacks.pgd import PGD
from torch.autograd import Variable
from contextlib import contextmanager
import numpy as np
from utils.context import ctx_noparamgrad_and_eval

# getv the gradient of parameters
def get_g(model):
    grad = [copy.deepcopy(p.grad.data.view(-1)) for n, p in model.named_parameters() if p.grad is not None]
    grad = torch.concat(grad)
    return grad

# load projected gradient to the model
def load_proj_g(model,grad_proj):
    count = 0 
    for n, p in model.named_parameters():
        n_param = p.numel()
        if p.grad is not None:
            p.grad.data.copy_(grad_proj[count:count+n_param].view_as(p))
        count += n_param

# get the gradients of a single batch
def get_gradients_per_batch(args, model, criterion, imgs, imgs_adv, labels):
    """
    Compute reference gradient on memory sample.
    """
    model.zero_grad()
    out = model(imgs)
    loss = criterion(out,labels)
    loss.backward()
    clean_gradients = [copy.deepcopy(p.grad.data.view(-1)) for n, p in model.named_parameters() if p.grad is not None]
    clean_gradients = torch.concat(clean_gradients)

    model.zero_grad()
    out_adv = model(imgs_adv.detach())
    loss = criterion(out_adv,labels)
    loss.backward()
    # torch.nn.utils.clip_grad_value_(model.parameters(),1)
    adv_gradients = [copy.deepcopy(p.grad.view(-1)) for n, p in model.named_parameters() if p.grad is not None]
    adv_gradients = torch.concat(adv_gradients)

    return clean_gradients,adv_gradients



def proj_g_orth(current_gradients, reference_gradients,model,threshold=0.5):

    """
    -current: adv gradient
    -reference: clean gradient
    """
    cos = F.cosine_similarity(current_gradients, reference_gradients)

    dotg = torch.dot(current_gradients, reference_gradients)
    norm = torch.sqrt(torch.dot(reference_gradients,reference_gradients)*torch.dot(current_gradients,current_gradients))
    dotg = dotg/norm


    # # clean to adver -> calibration to performance
    # proj_cur = current_gradients - torch.dot(current_gradients, reference_gradients) / torch.dot(reference_gradients,
    #                         reference_gradients)
    
    # # adver to clean -> calibration to robustness
    # proj_ref = reference_gradients - torch.dot(current_gradients, reference_gradients) / torch.dot(current_gradients,
    #                         current_gradients)

    
    # if dotg < 0:
    #     alpha2 = dotg / torch.dot(reference_gradients,
    #                         reference_gradients)
    #     grad_proj = current_gradients - \
    # reference_gradients * alpha2
    # else:
    #     grad_proj = current_gradients

    grad_proj_orth = current_gradients - reference_gradients*(torch.dot(current_gradients, reference_gradients)/torch.dot(reference_gradients,reference_gradients))
    return grad_proj_orth, dotg

def proj_g_thres(current_gradients, reference_gradients,model,threshold=0.5):

    """
    -current: adv gradient
    -reference: clean gradient
    """
    # cos = F.cosine_similarity(current_gradients, reference_gradients)
    # threshold = torch.Tensor(threshold)
    dotg = torch.dot(current_gradients, reference_gradients)
    norm = torch.sqrt(torch.dot(reference_gradients,reference_gradients)*torch.dot(current_gradients,current_gradients))
    dotg = dotg/norm
    cos = dotg

    ref_norm = torch.sqrt(torch.dot(reference_gradients,reference_gradients))
    cur_norm = torch.sqrt(torch.dot(current_gradients,current_gradients))

    if cos>=threshold:
        grad_proj = reference_gradients
    else:
        fac = (cur_norm*(threshold*torch.sqrt(1-cos**2)-cos*math.sqrt(1-threshold**2)))/(ref_norm*math.sqrt(1-threshold**2))
        grad_proj = current_gradients + fac*reference_gradients
    return grad_proj,dotg


def average_g(current_gradients, reference_gradients,model,lamda):
    dotg = torch.dot(current_gradients, reference_gradients)
    norm = torch.sqrt(torch.dot(reference_gradients,reference_gradients)*torch.dot(current_gradients,current_gradients))
    dotg = dotg/norm

    # print(dotg)

    return (1-lamda)*current_gradients + lamda*reference_gradients,dotg


# get refer and curr gradients
def gradient_adv_train(args, model, criterion, optimizer, imgs, imgs_adv, labels, epoch):
    """
    Compute reference gradient on memory sample.
    """
    model.zero_grad()
    out = model(imgs)
    loss = criterion(out,labels)
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    # torch.nn.utils.clip_grad_value_(model.parameters(),1)
    if args.grad_norm:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    clean_gradients = [copy.deepcopy(p.grad.data.view(-1)) for n, p in model.named_parameters() if p.grad is not None]
    clean_gradients = torch.concat(clean_gradients)

    model.zero_grad()
    out_adv = model(imgs_adv.detach())
    
    if args.use_trades:
        loss = trades_loss(model, imgs, labels, optimizer, epsilon=args.eps/255, perturb_steps=args.steps)
    else:
        loss = criterion(out_adv,labels)
    loss.backward()
    if args.grad_norm:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    # torch.nn.utils.clip_grad_value_(model.parameters(),1)
    adv_gradients = [copy.deepcopy(p.grad.view(-1)) for n, p in model.named_parameters() if p.grad is not None]
    adv_gradients = torch.concat(adv_gradients)
    
    if args.grad_proj == 'orth':
        grad_proj,dotg = proj_g_orth(current_gradients=adv_gradients, reference_gradients=clean_gradients,model=model)
    elif args.grad_proj == 'avg':
        grad_proj,dotg = average_g(current_gradients=adv_gradients, reference_gradients=clean_gradients, model=model, lamda=args.Lambda)
    elif args.grad_proj == 'thres':
        grad_proj,dotg = proj_g_thres(current_gradients=adv_gradients, reference_gradients=clean_gradients, model=model,threshold=args.grad_thres)
    else:
        raise ValueError('No such Grad Proj Method')
    model.zero_grad()

    load_proj_g(model, grad_proj=grad_proj)

    # print('adv norm',str(torch.norm(adv_gradients)))
    # print('clean norm',str(torch.norm(clean_gradients)))
    # print('Dif Ratio: ',str(grad_proj[grad_proj != clean_gradients].shape[0]/grad_proj.shape[0]))
    torch.autograd.set_detect_anomaly(True)
    # if grad_proj[grad_proj != clean_gradients].shape[0]/grad_proj.shape[0]>0:
    #     pdb.set_trace()

    # load_proj_g(model, grad_check)
    # grad_check_2 = [p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None]
    # grad_check_2 = torch.concat(grad_check_2)

    # pdb.set_trace()
    
    optimizer.step()

    return model,optimizer


# gradient track
def track_gradient(args, model, criterion, imgs, imgs_adv, labels, grad_track):
    clean_gradients,adv_gradients = get_gradients_per_batch(args, model, criterion, imgs, imgs_adv, labels)
    
    grad_track['Cos'] += F.cosine_similarity(clean_gradients.reshape([1,-1]),adv_gradients.reshape([1,-1]))
    grad_track['Eucli'] += torch.dist(clean_gradients,adv_gradients,p=2)
    grad_track['Clean_Norm'] += torch.norm(clean_gradients)  
    grad_track['Adver_Norm'] += torch.norm(adv_gradients)
    grad_track['L0_Ratio'] += torch.norm(clean_gradients-adv_gradients,p=0) / clean_gradients.shape[0]
    # pdb.set_trace()
    grad_track['KL'] += F.kl_div(clean_gradients,adv_gradients)
    return grad_track


# Gradient Surgery and Curriculumn Adver Training
def gradient_adv_train_cat(args, model, criterion, optimizer, imgs, imgs_adv, labels, epoch):
    """
    Compute reference gradient on memory sample.
    """
    
    k = args.k_cat
    each_batch = int(imgs.size(0) / (k+1))
    step_size = (args.steps / k)*np.random.uniform(1,10.0/k)
    alpha = min(args.eps * 1.25, args.eps + 4/255)/k*np.random.uniform(1,10.0/k)
    imgs_adv = imgs.clone()
    labels_adv = labels.clone()

    # Generate CAT Adver Samples with Batch Mixing
    for cur_k in range(k+1):
        attacker = PGD(eps=args.eps/255, steps=cur_k, alpha = alpha)

        if cur_k!=k:
            imgs_now, labels_now = imgs[cur_k*each_batch:(cur_k+1)*each_batch], labels[cur_k*each_batch:(cur_k+1)*each_batch]
        else:
            imgs_now, labels_now = imgs[k*each_batch:], labels[k*each_batch:]  # this gets max share
        
        with ctx_noparamgrad_and_eval(model):
            imgs_now_adv = attacker.attack(model,imgs_now,labels_now) if cur_k!=0 else imgs_now
        
        if cur_k!=k:
            imgs_adv[cur_k*each_batch:(cur_k+1)*each_batch], labels_adv[cur_k*each_batch:(cur_k+1)*each_batch] = imgs_now_adv, labels_now
        else:
            imgs_adv[k*each_batch:], labels_adv[k*each_batch:]= imgs_now_adv, labels_now
    

    model.zero_grad()
    out = model(imgs)
    loss = criterion(out,labels)
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    # torch.nn.utils.clip_grad_value_(model.parameters(),1)
    if args.grad_norm:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    clean_gradients = [copy.deepcopy(p.grad.data.view(-1)) for n, p in model.named_parameters() if p.grad is not None]
    clean_gradients = torch.concat(clean_gradients)

    # model.zero_grad()
    # out = model(imgs)
    # loss = criterion(out,labels)
    # loss.backward()
    # # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    # # torch.nn.utils.clip_grad_value_(model.parameters(),1)
    # clean_gradients_2 = [copy.deepcopy(p.grad.data.view(-1)) for n, p in model.named_parameters() if p.grad is not None]
    # clean_gradients_2 = torch.concat(clean_gradients_2)

    model.zero_grad()
    out_adv = model(imgs_adv.detach())
    
    if args.use_trades:
        loss = trades_loss(model, imgs, labels, optimizer, epsilon=args.eps/255, perturb_steps=args.steps)
    else:
        loss = criterion(out_adv,labels)
    loss.backward()
    if args.grad_norm:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    
    # torch.nn.utils.clip_grad_value_(model.parameters(),1)
    adv_gradients = [copy.deepcopy(p.grad.view(-1)) for n, p in model.named_parameters() if p.grad is not None]
    adv_gradients = torch.concat(adv_gradients)
    
    if args.grad_proj == 'orth':
        grad_proj,dotg = proj_g_orth(current_gradients=adv_gradients, reference_gradients=clean_gradients,model=model)
    elif args.grad_proj == 'avg':
        grad_proj,dotg = average_g(current_gradients=adv_gradients, reference_gradients=clean_gradients, model=model, lamda=args.Lambda)
    elif args.grad_proj == 'thres':
        grad_proj,dotg = proj_g_thres(current_gradients=adv_gradients, reference_gradients=clean_gradients, model=model,threshold=args.grad_thres)
    else:
        raise ValueError('No such Grad Proj Method')
    model.zero_grad()
    # print(dotg)
    load_proj_g(model, grad_proj=grad_proj)

    # print('adv norm',str(torch.norm(adv_gradients)))
    # print('clean norm',str(torch.norm(clean_gradients)))
    # print('Dif Ratio: ',str(grad_proj[grad_proj != clean_gradients].shape[0]/grad_proj.shape[0]))
    torch.autograd.set_detect_anomaly(True)
    # if grad_proj[grad_proj != clean_gradients].shape[0]/grad_proj.shape[0]>0:
    #     pdb.set_trace()

    # load_proj_g(model, grad_check)
    # grad_check_2 = [p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None]
    # grad_check_2 = torch.concat(grad_check_2)

    # pdb.set_trace()
    
    optimizer.step()

    return model,optimizer

