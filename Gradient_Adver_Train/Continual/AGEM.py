from webbrowser import get
import torch
import torch.nn.functional as F
from utils.context import ctx_noparamgrad_and_eval


import torch
from torch.utils.data import random_split

from avalanche.benchmarks.utils.data_loader import \
    GroupBalancedInfiniteDataLoader
from avalanche.models import avalanche_forward
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
import pdb

# After Training a Exp
@torch.no_grad()
def update_memory(dataset,patterns_per_experience,buffers,sample_size):
    """
    Update replay memory with patterns from current experience.
    """
    removed_els = len(dataset) - patterns_per_experience
    if removed_els > 0:
        dataset, _ = random_split(dataset,
                                    [patterns_per_experience,
                                    removed_els])
    buffers.append(dataset)
    buffer_dataloader = GroupBalancedInfiniteDataLoader(
        buffers,
        batch_size=sample_size // len(buffers),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True)
    buffer_dliter = iter(buffer_dataloader)

    return buffers,buffer_dliter,buffer_dataloader


def sample_from_memory(buffer_dliter):
    """
    Sample a minibatch from memory.
    Return a tuple of patterns (tensor), targets (tensor).
    """
    return next(buffer_dliter)

# return g_ref (clean and adv)
def get_ref_g(buffers, model, optimizer, buffer_dliter, criterion, attacker):
    """
    Compute reference gradient on memory sample.
    """
    if len(buffers) > 0:
        model.train()
        optimizer.zero_grad()
        mb = sample_from_memory(buffer_dliter)
        xref, yref = mb[0], mb[1]
        xref, yref = xref.cuda(), yref.cuda()

        with ctx_noparamgrad_and_eval(model):
            xref_adv = attacker.attack(model, xref, yref)
        
        # clean data
        out = model(xref)
        loss = criterion(out,yref)
        loss.backward()
        # gradient can be None for some head on multi-headed models
        reference_gradients = [
            p.grad.view(-1) if p.grad is not None
            else torch.zeros(p.numel()).cuda()
            for n, p in model.named_parameters()]
        
        reference_gradients = torch.cat(reference_gradients)
        optimizer.zero_grad()

        # adver data
        out = model(xref_adv)
        loss = criterion(out,yref)
        loss.backward()
        # gradient can be None for some head on multi-headed models
        reference_gradients_adv = [
            p.grad.view(-1) if p.grad is not None
            else torch.zeros(p.numel()).cuda()
            for n, p in model.named_parameters()]
        
        reference_gradients_adv = torch.cat(reference_gradients_adv)
        optimizer.zero_grad()

        return reference_gradients, reference_gradients_adv
    else:
        return None, None


def g_cur_return(model, reference_gradients):

    current_gradients = [
        p.grad.view(-1) if p.grad is not None
        else torch.zeros(p.numel()).cuda()
        for n, p in model.named_parameters()]
    current_gradients = torch.cat(current_gradients)

    assert current_gradients.shape == reference_gradients.shape, \
        "Different model parameters in AGEM projection"
    
    return current_gradients

def proj_g(current_gradients, reference_gradients):
    dotg = torch.dot(current_gradients, reference_gradients)
    # print('dotg ',dotg)
    # pdb.set_trace()
    if dotg < 0:
        alpha2 = dotg / torch.dot(reference_gradients,
                                    reference_gradients)
        grad_proj = current_gradients - \
            reference_gradients * alpha2
    else:
        grad_proj = current_gradients
    return grad_proj

def load_proj_g(model,grad_proj):
    count = 0 
    for n, p in model.named_parameters():
        n_param = p.numel()
        if p.grad is not None:
            p.grad.copy_(grad_proj[count:count+n_param].view_as(p))
        count += n_param

def AGEM(train_loader,model,attacker,optimizer,args,scheduler,accs,accs_adv,losses,buffers,buffer_dliter,criterion):
    print('len(buffers) ',len(buffers))
    if len(buffers) == 0:
        for i, data in enumerate(train_loader):
            imgs, labels = data[0],data[1]
            # labels = label_transform(labels,experience.classes_in_this_experience)
            # print(labels)
            imgs, labels = imgs.cuda(), labels.cuda()
            # generate adversarial images:
            if args.Lambda != 0:
                with ctx_noparamgrad_and_eval(model):
                    imgs_adv = attacker.attack(model, imgs, labels)
                logits_adv = model(imgs_adv.detach())
            # logits for clean imgs:
            logits = model(imgs)

            # loss and update:
            loss = F.cross_entropy(logits, labels)
            if args.Lambda != 0:
                loss = (1-args.Lambda) * loss + args.Lambda * F.cross_entropy(logits_adv, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_lr = scheduler.get_lr()[0]
            # metrics:
            accs.append((logits.argmax(1) == labels).float().mean().item())
            if args.Lambda != 0:
                accs_adv.append((logits_adv.argmax(1) == labels).float().mean().item())
            losses.append(loss.item())
    else:
        for i, data in enumerate(train_loader):
            reference_gradients, reference_gradients_adv = get_ref_g(buffers, model, optimizer, buffer_dliter, criterion, attacker)
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
            
            # mode 'avg_before'
            if args.agem_mode == 'avg_before':
                
                # avg standard gradients and adver gradients
                reference_gradients = (1-args.Lambda) * reference_gradients + args.Lambda * reference_gradients_adv

                loss = F.cross_entropy(logits, labels)
                if args.Lambda != 0:
                    loss = (1-args.Lambda) * loss + args.Lambda * F.cross_entropy(logits_adv, labels)

                current_lr = scheduler.get_lr()[0]
                # metrics:
                accs.append((logits.argmax(1) == labels).float().mean().item())
                if args.Lambda != 0:
                    accs_adv.append((logits_adv.argmax(1) == labels).float().mean().item())
                else:
                    accs_adv.append((logits_adv.argmax(1) == labels).float().mean().item())

                optimizer.zero_grad()
                loss.backward()

                current_gradients = g_cur_return(model, reference_gradients)
                grad_proj = proj_g(current_gradients, reference_gradients)
                load_proj_g(model,grad_proj)
                # pdb.set_trace()
                optimizer.step()

                losses.append(loss.item())
            elif args.agem_mode == 'avg_after':

                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                current_gradients = g_cur_return(model, reference_gradients)
                grad_proj = proj_g(current_gradients, reference_gradients)
                # load_proj_g(model,grad_proj)
                optimizer.zero_grad()

                loss_adv = F.cross_entropy(logits_adv, labels)
                optimizer.zero_grad()
                loss_adv.backward()
                current_gradients_adv = g_cur_return(model, reference_gradients_adv)
                grad_proj_adv = proj_g(current_gradients_adv, reference_gradients_adv)
                optimizer.zero_grad()

                grad_proj = (1-args.Lambda)*grad_proj + args.Lambda*grad_proj_adv
                load_proj_g(model,grad_proj)
                optimizer.step()

                losses.append(loss.item())
            else:
                raise ValueError('No such '+args.agem_mode)