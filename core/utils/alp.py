import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from core.metrics import accuracy


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()

def clp_loss(model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, 
                attack='linf-pgd'):
    """
    ALP Training
    """
    model.eval()
    batch_size = len(x_natural)
    
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if attack == 'linf-pgd':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        raise ValueError(f'Attack={attack} not supported for MART training!')
    model.train()
    
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits = model(x_natural)
    logits_adv = model(x_adv)

    adv_probs = F.softmax(logits_adv, dim=1)
    probs = F.softmax(logits,dim=1)

    loss_robust = torch.norm(adv_probs-probs,p=2,dim=-1).mean()
    
    batch_metrics = {'adv_loss': loss_robust.item(), 'adv_acc': accuracy(y, logits_adv.detach())}
    return loss_robust, batch_metrics
