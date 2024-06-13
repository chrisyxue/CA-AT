import pdb
import torch
import torch.nn.functional as F
from utils.context import ctx_noparamgrad_and_eval

from avalanche.models import avalanche_forward, MultiTaskModule
from avalanche.training.plugins.strategy_plugin import StrategyPlugin



def distillation_loss(out, prev_out,temperature=1):
    """
    Compute distillation loss between output of the current model and
    and output of the previous (saved) model.
    """
    # we compute the loss only on the previously active units.
    log_p = torch.log_softmax(out / temperature, dim=1)
    q = torch.softmax(prev_out / temperature, dim=1)
    res = torch.nn.functional.kl_div(log_p, q, reduction='batchmean')
    return res

def penalty(out, x, alpha, model, prev_model, tempreture):
    """
    Compute weighted distillation loss.
    """

    with torch.no_grad():
        y_prev = prev_model(x)
        y_curr = out
    dist_loss = distillation_loss(y_curr, y_prev, tempreture)
    return alpha * dist_loss

def LwF(train_loader,model,attacker,optimizer,args,scheduler,accs,accs_adv,losses,prev_model):
    if prev_model is None:
        print('prev_model is None')
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
        tempreture = args.tempreture
        alpha = args.alpha
        for i, data in enumerate(train_loader):
            imgs, labels = data[0],data[1]
            # labels = label_transform(labels,experience.classes_in_this_experience)
            # print(labels)
            imgs, labels = imgs.cuda(), labels.cuda()
            # generate adversarial images:
            # if args.Lambda != 0:
            with ctx_noparamgrad_and_eval(model):
                imgs_adv = attacker.attack(model, imgs, labels)
            logits_adv = model(imgs_adv.detach())
            # logits for clean imgs:
            logits = model(imgs)

            # loss and update:
            loss = F.cross_entropy(logits, labels)
            p = penalty(logits, imgs, alpha, model, prev_model, tempreture)
            # pdb.set_trace()
            # loss = loss + p
            loss = p

            loss_adv = F.cross_entropy(logits_adv, labels)
            p_adv = penalty(logits, imgs_adv, alpha, model, prev_model, tempreture)
            loss_adv = loss_adv + p

            if args.Lambda != 0:
                loss = (1-args.Lambda) * loss + args.Lambda * loss_adv
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_lr = scheduler.get_lr()[0]
            # metrics:
            accs.append((logits.argmax(1) == labels).float().mean().item())
            if args.Lambda != 0:
                accs_adv.append((logits_adv.argmax(1) == labels).float().mean().item())
            else:
                accs_adv.append((logits_adv.argmax(1) == labels).float().mean().item())
            losses.append(loss.item())
    return train_loader,model,attacker,optimizer,args,scheduler,accs,accs_adv,losses,prev_model




