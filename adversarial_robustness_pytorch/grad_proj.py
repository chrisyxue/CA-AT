import torch
import torch.nn.functional as F
import torch
from torch.utils.data import random_split
import copy
import pandas as pd
import math
import sys

# from attacks.pgd import PGD
from torch.autograd import Variable
from contextlib import contextmanager
import numpy as np
import pdb
import torch.nn as nn

# getv the gradient of parameters
def get_g(model):
    grad = [copy.deepcopy(p.grad.data.view(-1)) for n, p in model.named_parameters() if p.grad is not None]
    grad = torch.concat(grad)
    layer_names = [n for n, p in model.named_parameters() if p.grad is not None]
    layer_grad_dims = [p.grad.data.view(-1).shape[0] for n, p in model.named_parameters() if p.grad is not None]
    return layer_names, layer_grad_dims, grad

# load projected gradient to the model
def load_proj_g(model,grad):

    # for numerical stable
    grad = torch.nan_to_num(grad)
    for param in model.parameters():
            if param.grad is None:
                param.grad = torch.zeros_like(param)
    count = 0 
    for n, p in model.named_parameters():
        n_param = p.numel()
        if p.grad is not None:
            p.grad.data.copy_(grad[count:count+n_param].view_as(p))
        count += n_param

# get the gradients of a single batch
def get_gradients_per_batch(model, clean_loss, adv_loss):
    """
    Compute reference gradient on memory sample.
    """
    clean_loss.backward()
    clean_layer_names, clean_layer_grad_dims,clean_gradients = get_g(model)
    model.zero_grad()

    adv_loss.backward()
    adv_gradients = get_g(model)
    adv_layer_names, adv_layer_grad_dims,adv_gradients = get_g(model)
    model.zero_grad()

    return clean_layer_names, clean_layer_grad_dims,clean_gradients,adv_layer_names, adv_layer_grad_dims,adv_gradients

# track the properties of adv and clean gradient
def track_gradient(clean_gradients,adv_gradients):
    grad_track = {}
    grad_track['Cos'] = F.cosine_similarity(clean_gradients.reshape([1,-1]),adv_gradients.reshape([1,-1])).detach().item()
    grad_track['Eucli'] = torch.dist(clean_gradients,adv_gradients,p=2).detach().item()
    grad_track['Clean_L2Norm'] = torch.norm(clean_gradients,p=2).detach().item()  
    grad_track['Adver_L2Norm'] = torch.norm(adv_gradients,p=2).detach().item()
    grad_track['Clean_L1Norm'] = torch.norm(clean_gradients,p=1).detach().item()  
    grad_track['Adver_L1Norm'] = torch.norm(adv_gradients,p=1).detach().item()
    grad_track['Conflict'] = grad_track['Clean_L2Norm']*grad_track['Adver_L2Norm']*(1-grad_track['Cos'])
    grad_track['Dim'] = clean_gradients.shape[0]
    return grad_track

# track the properties of adv and clean gradient per layer
def track_gradient_per_layer(clean_layer_names,clean_layer_grad_dims,clean_gradients,adv_layer_names,adv_layer_grad_dims,adv_gradients):
    grad_track_layer = {}
    for i in range(len(clean_layer_names)):
        layer_name = clean_layer_names[i]
        if i==0:
            clean_g_layer = clean_gradients[:clean_layer_grad_dims[i]]
            adv_g_layer = adv_gradients[:adv_layer_grad_dims[i]]
        else:
            start = sum(clean_layer_grad_dims[:i])
            clean_g_layer = clean_gradients[start:start+clean_layer_grad_dims[i]]
            adv_g_layer = adv_gradients[start:start+adv_layer_grad_dims[i]]
        grad_track = track_gradient(clean_g_layer,adv_g_layer)

        grad_track_layer[layer_name] = grad_track 
    
    grad_track_layer = pd.DataFrame(grad_track_layer)
    return grad_track_layer 

def proj_g_orth(current_gradients, reference_gradients):

    """
    -current: adv gradient
    -reference: clean gradient
    """
    cos = F.cosine_similarity(current_gradients, reference_gradients)

    dotg = torch.dot(current_gradients, reference_gradients)
    norm = torch.sqrt(torch.dot(reference_gradients,reference_gradients)*torch.dot(current_gradients,current_gradients))
    dotg = dotg/norm

    grad_proj_orth = current_gradients - reference_gradients*(torch.dot(current_gradients, reference_gradients)/torch.dot(reference_gradients,reference_gradients))
    return grad_proj_orth, dotg



def proj_g_thres(current_gradients, reference_gradients, threshold=0.5):

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

def proj_g_thres_layer(current_gradients, reference_gradients, layer_grad_dims, threshold=0.5):

    grad_proj = []
    for i in range(len(layer_grad_dims)):
        if i==0:
            current_g_layer = current_gradients[:layer_grad_dims[i]]
            reference_g_layer = reference_gradients[:layer_grad_dims[i]]
        else:
            start = sum(layer_grad_dims[:i])
            current_g_layer = current_gradients[start:start+layer_grad_dims[i]]
            reference_g_layer = reference_gradients[start:start+layer_grad_dims[i]]
        grad_proj_layer,dotg = proj_g_thres(current_gradients=current_g_layer, reference_gradients=reference_g_layer,threshold=threshold)
        grad_proj.append(grad_proj_layer)
    grad_proj = torch.cat(grad_proj)
    return grad_proj,dotg

def average_g(current_gradients, reference_gradients,beta):
    """
    -current: adv gradient
    -reference: clean gradient
    """
    dotg = torch.dot(current_gradients, reference_gradients)
    norm = torch.sqrt(torch.dot(reference_gradients,reference_gradients)*torch.dot(current_gradients,current_gradients))
    dotg = dotg/norm
    return beta*current_gradients + (1-beta)*reference_gradients,dotg


# def gradient_adv_train_layer_wise(args, model, clean_loss, adv_loss, optimizer):
#     optimizer.zero_grad()
#     clean_layer_names, clean_layer_grad_dims,clean_gradients,adv_layer_names, adv_layer_grad_dims,adv_gradients = get_gradients_per_batch(model, clean_loss, adv_loss)
    
#     # Operations & Optimization
#     if args.grad_norm:
#         adv_gradients = adv_gradients / torch.norm(adv_gradients,p=2)
#         clean_gradients = clean_gradients / torch.norm(clean_gradients,p=2)

#     if args.grad_op == 'orth':
#         grad,dotg = proj_g_orth(current_gradients=adv_gradients, reference_gradients=clean_gradients,model=model)
#     elif args.grad_op == 'avg':
#         grad,dotg = average_g(current_gradients=adv_gradients, reference_gradients=clean_gradients, beta=args.beta)
#     elif args.grad_op == 'thres':
#         grad,dotg = proj_g_thres(current_gradients=adv_gradients, reference_gradients=clean_gradients,threshold=args.grad_thres)
#     elif args.grad_op == 'thres_layer':
#         # Only project the gradient with respect to each shortcut
#         grad,dotg = proj_g_thres_layer(current_gradients=adv_gradients, reference_gradients=clean_gradients, layer_grad_dims=clean_layer_grad_dims, threshold=args.grad_thres)
#     else:
#         raise ValueError('No such Grad Op Method')



def thres_update(args,epoch,clean_metrics,adv_metrics):
    """
    adaptively update the thres for gradient surgery
    """
    if args.adaptive_thres == 'No':
        return args
    elif args.adaptive_thres == 'StdAccLin':
        std_acc = clean_metrics['clean_acc']
        args.grad_thres = 1 - std_acc + 1e-3
        return args
    elif args.adaptive_thres == 'StdAccExp':
        std_acc = clean_metrics['clean_acc']
        args.grad_thres = math.exp(1 - std_acc + 1e-3) / math.exp(1)
        return args
    else:
        raise ValueError('No such thres adaptive strategy')
    
def gradient_adv_train(args, model, clean_loss, adv_loss, optimizer):
    optimizer.zero_grad()
    clean_layer_names, clean_layer_grad_dims,clean_gradients,adv_layer_names, adv_layer_grad_dims,adv_gradients = get_gradients_per_batch(model, clean_loss, adv_loss)

    # Operations & Optimization
    if args.grad_norm:
        adv_gradients = adv_gradients / torch.norm(adv_gradients,p=2)
        clean_gradients = clean_gradients / torch.norm(clean_gradients,p=2)

    if args.grad_op == 'orth':
        grad,dotg = proj_g_orth(current_gradients=adv_gradients, reference_gradients=clean_gradients,model=model)
    elif args.grad_op == 'avg':
        grad,dotg = average_g(current_gradients=adv_gradients, reference_gradients=clean_gradients, beta=args.beta)
    elif args.grad_op == 'thres':
        grad,dotg = proj_g_thres(current_gradients=adv_gradients, reference_gradients=clean_gradients,threshold=args.grad_thres)
    elif args.grad_op == 'thres_layer':
        # Only project the gradient with respect to each shortcut
        grad,dotg = proj_g_thres_layer(current_gradients=adv_gradients, reference_gradients=clean_gradients, layer_grad_dims=clean_layer_grad_dims, threshold=args.grad_thres)
    else:
        raise ValueError('No such Grad Op Method')
    load_proj_g(model,grad)
    # pdb.set_trace()
    if args.clip_grad:
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    optimizer.step()
    

# def gradient_track(args, model, clean_loss, adv_loss, optimizer):
#     clean_layer_names, clean_layer_grad_dims,clean_gradients,adv_layer_names, adv_layer_grad_dims,adv_gradients = get_gradients_per_batch(model, clean_loss, adv_loss)
#     grad_track = 
   





    


    
    
    




# # get refer and curr gradients
# def gradient_adv_train(args, model, criterion, optimizer, imgs, imgs_adv, labels, epoch):
#     """
#     Compute reference gradient on memory sample.
#     """
#     model.zero_grad()
#     out = model(imgs)
#     loss = criterion(out,labels)
#     loss.backward()
#     # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
#     # torch.nn.utils.clip_grad_value_(model.parameters(),1)
#     if args.grad_norm:
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
#     clean_gradients = [copy.deepcopy(p.grad.data.view(-1)) for n, p in model.named_parameters() if p.grad is not None]
#     clean_gradients = torch.concat(clean_gradients)

#     model.zero_grad()
#     out_adv = model(imgs_adv.detach())
    
#     if args.use_trades:
#         loss = trades_loss(model, imgs, labels, optimizer, epsilon=args.eps/255, perturb_steps=args.steps)
#     else:
#         loss = criterion(out_adv,labels)
#     loss.backward()
#     if args.grad_norm:
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
#     # torch.nn.utils.clip_grad_value_(model.parameters(),1)
#     adv_gradients = [copy.deepcopy(p.grad.view(-1)) for n, p in model.named_parameters() if p.grad is not None]
#     adv_gradients = torch.concat(adv_gradients)
    
#     if args.grad_proj == 'orth':
#         grad_proj,dotg = proj_g_orth(current_gradients=adv_gradients, reference_gradients=clean_gradients,model=model)
#     elif args.grad_proj == 'avg':
#         grad_proj,dotg = average_g(current_gradients=adv_gradients, reference_gradients=clean_gradients, model=model, lamda=args.Lambda)
#     elif args.grad_proj == 'thres':
#         grad_proj,dotg = proj_g_thres(current_gradients=adv_gradients, reference_gradients=clean_gradients, model=model,threshold=args.grad_thres)
#     else:
#         raise ValueError('No such Grad Proj Method')
#     model.zero_grad()

#     load_proj_g(model, grad_proj=grad_proj)

#     # print('adv norm',str(torch.norm(adv_gradients)))
#     # print('clean norm',str(torch.norm(clean_gradients)))
#     # print('Dif Ratio: ',str(grad_proj[grad_proj != clean_gradients].shape[0]/grad_proj.shape[0]))
#     torch.autograd.set_detect_anomaly(True)
#     # if grad_proj[grad_proj != clean_gradients].shape[0]/grad_proj.shape[0]>0:
#     #     pdb.set_trace()

#     # load_proj_g(model, grad_check)
#     # grad_check_2 = [p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None]
#     # grad_check_2 = torch.concat(grad_check_2)

#     # pdb.set_trace()
    
#     optimizer.step()

#     return model,optimizer


# # gradient track
# def track_gradient(args, model, criterion, imgs, imgs_adv, labels, grad_track):
#     clean_gradients,adv_gradients = get_gradients_per_batch(args, model, criterion, imgs, imgs_adv, labels)
    
#     grad_track['Cos'] += F.cosine_similarity(clean_gradients.reshape([1,-1]),adv_gradients.reshape([1,-1]))
#     grad_track['Eucli'] += torch.dist(clean_gradients,adv_gradients,p=2)
#     grad_track['Clean_Norm'] += torch.norm(clean_gradients)  
#     grad_track['Adver_Norm'] += torch.norm(adv_gradients)
#     grad_track['L0_Ratio'] += torch.norm(clean_gradients-adv_gradients,p=0) / clean_gradients.shape[0]
#     # pdb.set_trace()
#     grad_track['KL'] += F.kl_div(clean_gradients,adv_gradients)
#     return grad_track


# # Gradient Surgery and Curriculumn Adver Training
# def gradient_adv_train_cat(args, model, criterion, optimizer, imgs, imgs_adv, labels, epoch):
#     """
#     Compute reference gradient on memory sample.
#     """
    
#     k = args.k_cat
#     each_batch = int(imgs.size(0) / (k+1))
#     step_size = (args.steps / k)*np.random.uniform(1,10.0/k)
#     alpha = min(args.eps * 1.25, args.eps + 4/255)/k*np.random.uniform(1,10.0/k)
#     imgs_adv = imgs.clone()
#     labels_adv = labels.clone()

#     # Generate CAT Adver Samples with Batch Mixing
#     for cur_k in range(k+1):
#         attacker = PGD(eps=args.eps/255, steps=cur_k, alpha = alpha)

#         if cur_k!=k:
#             imgs_now, labels_now = imgs[cur_k*each_batch:(cur_k+1)*each_batch], labels[cur_k*each_batch:(cur_k+1)*each_batch]
#         else:
#             imgs_now, labels_now = imgs[k*each_batch:], labels[k*each_batch:]  # this gets max share
        
#         with ctx_noparamgrad_and_eval(model):
#             imgs_now_adv = attacker.attack(model,imgs_now,labels_now) if cur_k!=0 else imgs_now
        
#         if cur_k!=k:
#             imgs_adv[cur_k*each_batch:(cur_k+1)*each_batch], labels_adv[cur_k*each_batch:(cur_k+1)*each_batch] = imgs_now_adv, labels_now
#         else:
#             imgs_adv[k*each_batch:], labels_adv[k*each_batch:]= imgs_now_adv, labels_now
    

#     model.zero_grad()
#     out = model(imgs)
#     loss = criterion(out,labels)
#     loss.backward()
#     # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
#     # torch.nn.utils.clip_grad_value_(model.parameters(),1)
#     if args.grad_norm:
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
#     clean_gradients = [copy.deepcopy(p.grad.data.view(-1)) for n, p in model.named_parameters() if p.grad is not None]
#     clean_gradients = torch.concat(clean_gradients)

#     # model.zero_grad()
#     # out = model(imgs)
#     # loss = criterion(out,labels)
#     # loss.backward()
#     # # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
#     # # torch.nn.utils.clip_grad_value_(model.parameters(),1)
#     # clean_gradients_2 = [copy.deepcopy(p.grad.data.view(-1)) for n, p in model.named_parameters() if p.grad is not None]
#     # clean_gradients_2 = torch.concat(clean_gradients_2)

#     model.zero_grad()
#     out_adv = model(imgs_adv.detach())
    
#     if args.use_trades:
#         loss = trades_loss(model, imgs, labels, optimizer, epsilon=args.eps/255, perturb_steps=args.steps)
#     else:
#         loss = criterion(out_adv,labels)
#     loss.backward()
#     if args.grad_norm:
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    
#     # torch.nn.utils.clip_grad_value_(model.parameters(),1)
#     adv_gradients = [copy.deepcopy(p.grad.view(-1)) for n, p in model.named_parameters() if p.grad is not None]
#     adv_gradients = torch.concat(adv_gradients)
    
#     if args.grad_proj == 'orth':
#         grad_proj,dotg = proj_g_orth(current_gradients=adv_gradients, reference_gradients=clean_gradients,model=model)
#     elif args.grad_proj == 'avg':
#         grad_proj,dotg = average_g(current_gradients=adv_gradients, reference_gradients=clean_gradients, model=model, lamda=args.Lambda)
#     elif args.grad_proj == 'thres':
#         grad_proj,dotg = proj_g_thres(current_gradients=adv_gradients, reference_gradients=clean_gradients, model=model,threshold=args.grad_thres)
#     else:
#         raise ValueError('No such Grad Proj Method')
#     model.zero_grad()
#     # print(dotg)
#     load_proj_g(model, grad_proj=grad_proj)

#     # print('adv norm',str(torch.norm(adv_gradients)))
#     # print('clean norm',str(torch.norm(clean_gradients)))
#     # print('Dif Ratio: ',str(grad_proj[grad_proj != clean_gradients].shape[0]/grad_proj.shape[0]))
#     torch.autograd.set_detect_anomaly(True)
#     # if grad_proj[grad_proj != clean_gradients].shape[0]/grad_proj.shape[0]>0:
#     #     pdb.set_trace()

#     # load_proj_g(model, grad_check)
#     # grad_check_2 = [p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None]
#     # grad_check_2 = torch.concat(grad_check_2)

#     # pdb.set_trace()
    
#     optimizer.step()

#     return model,optimizer

