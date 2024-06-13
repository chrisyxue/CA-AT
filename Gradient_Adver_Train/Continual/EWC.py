from collections import defaultdict
from typing import Dict, Tuple
import warnings

import torch
from torch import Tensor
from torch.utils.data import DataLoader

import torch.nn.functional as F
from utils.context import ctx_noparamgrad_and_eval

from avalanche.models.utils import avalanche_forward
from avalanche.training.utils import copy_params_dict, zerolike_params_dict
import pdb

def compute_importances(model, criterion, optimizer,dataset, batch_size):
    """
    Compute EWC importance matrix for each parameter
    """

    model.eval()

    # list of list
    importances = zerolike_params_dict(model)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    # for i, batch in enumerate(dataloader):
    for i, data in enumerate(dataloader):    
        # get only input, target and task_id from the batch
        x, y = data[0], data[1]
        # x, y, task_labels = batch[0], batch[1], batch[-1]
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        # out = avalanche_forward(model, x, task_labels)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()

        for (k1, p), (k2, imp) in zip(model.named_parameters(),
                                        importances):
            assert (k1 == k2)
            if p.grad is not None:
                imp += p.grad.data.clone().pow(2)

    # average over mini batch length
    for _, imp in importances:
        imp /= float(len(dataloader))

    return importances

@torch.no_grad()
def update_importances(importances, t, importances_list, keep_importance_data, decay_factor, mode):
    """
    Update importance for each parameter based on the currently computed
    importances.

    t is the task id
    """
    if mode == 'separate' or t == 0:
        importances_list[t] = importances
    elif mode == 'online':
        for (k1, old_imp), (k2, curr_imp) in \
                zip(importances_list[t - 1], importances):
            assert k1 == k2, 'Error in importance computation.'
            # print(decay_factor)
            # print(old_imp)
            importances_list[t].append(
                (k1, (decay_factor * old_imp + curr_imp)))

        # clear previous parameter importances
        if t > 0 and (not keep_importance_data):
            del importances_list[t - 1]

    else:
        raise ValueError("Wrong EWC mode.")

def update_ewc_params_importance(experience, model, criterion, optimizer, batch_size, saved_params, keep_importance_data, 
                        exp_counter, decay_factor, importances_list, mode):
    """
    Compute importances of parameters after each experience.
    """
    # exp_counter = strategy.clock.train_exp_counter
    importances = compute_importances(model,
                                    criterion,
                                    optimizer,
                                    experience.dataset,
                                    batch_size)
    update_importances(importances, exp_counter, importances_list, keep_importance_data, decay_factor, mode)
    saved_params[exp_counter] = \
        copy_params_dict(model)
    
    # clear previous parameter values if keep_importance_data is False
    if exp_counter > 0 and \
            (not keep_importance_data):
        del saved_params[exp_counter - 1]


def before_backward(loss, model, exp_counter, mode, saved_params, ewc_lambda,
                    importances_list):
        """
        Compute EWC penalty and add it to the loss.
        """
        # exp_counter = strategy.clock.train_exp_counter
        if exp_counter == 0:
            return loss

        # penalty = torch.tensor(0).float().to(strategy.device)
        penalty = torch.tensor(0).float().cuda()

        if mode == 'separate':
            for experience_id in range(exp_counter):
                for (_, cur_param), (_, saved_param), (_, imp) in zip(
                        model.named_parameters(),
                        saved_params[experience_id],
                        importances_list[experience_id]):
                    penalty += (imp * (cur_param - saved_param).pow(2)).sum()
        elif mode == 'online':
            prev_exp = exp_counter - 1
            for (_, cur_param), (_, saved_param), (_, imp) in zip(
                    model.named_parameters(),
                    saved_params[prev_exp],
                    importances_list[prev_exp]):
                penalty += (imp * (cur_param - saved_param).pow(2)).sum()
        else:
            raise ValueError('Wrong EWC mode.')

        loss = loss + ewc_lambda * penalty
        # pdb.set_trace()
        # pdb.set_trace()
        return loss

def EWC(train_loader,model,attacker,optimizer,args,scheduler,accs,accs_adv,losses,saved_params,importances_list,exp_counter):
    
    ewc_lambda = args.ewc_lambda
    ewc_mode = args.ewc_mode
    ewc_decay_factor = args.ewc_decay_factor

    # pdb.set_trace()
    for i, data in enumerate(train_loader):
        imgs, labels = data[0],data[1]
        # labels = label_transform(labels,experience.classes_in_this_experience)
        # print(labels)
        imgs, labels = imgs.cuda(), labels.cuda()
        # generate adversarial images:
        
        with ctx_noparamgrad_and_eval(model):
            imgs_adv = attacker.attack(model, imgs, labels)
        logits_adv = model(imgs_adv.detach())
        # logits for clean imgs:
        logits = model(imgs)
        
        # loss and update:
        loss = F.cross_entropy(logits, labels)

        loss_adv = F.cross_entropy(logits_adv, labels)


        if args.Lambda != 0:
            loss = (1-args.Lambda) * loss + args.Lambda * loss_adv

        loss = before_backward(loss, model, exp_counter, ewc_mode, saved_params, ewc_lambda, importances_list)
        # pdb.set_trace()
        current_lr = scheduler.get_lr()[0]
        # metrics:
        accs.append((logits.argmax(1) == labels).float().mean().item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.Lambda != 0:
            accs_adv.append((logits_adv.argmax(1) == labels).float().mean().item())
        else:
            accs_adv.append((logits_adv.argmax(1) == labels).float().mean().item())
        losses.append(loss.item())