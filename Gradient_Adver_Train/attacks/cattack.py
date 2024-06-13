import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
import pdb
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


"""
Input: X,\Theta,t
Output: Diff(X), Diff(\Theta)
"""
class Adver_Op(nn.Module):
    def __init__(self,loss_fn,model,eps,lr=0.01):
        super(Adver_Op, self).__init__()
        self.model = model
        self.loss_fn = loss_fn if loss_fn else nn.CrossEntropyLoss(reduction="sum") 
        self.eps = eps
        self.param_keys = None
        self.lr = lr

    def get_param_keys(self,param_keys):
        self.param_keys = param_keys

    # Get Diff X
    def diff_x(self,model,x,labels):
        model.eval().cuda()

        # initialize x_adv:
        # x_adv = x.clone()
        # x_adv += (2.0 * torch.rand(x_adv.shape).cuda() - 1.0) * self.eps # random initialize
        # x_adv = torch.clamp(x_adv, 0, 1) # clamp to RGB range [0,1]
        # x_adv = Variable(x_adv.cuda(), requires_grad=True)

        # logits_adv = model(x_adv)
        # loss_adv = self.loss_fn(logits_adv, labels)
        # grad_adv = torch.autograd.grad(loss_adv, x_adv, only_inputs=True)[0]
        atk = torchattacks.PGD(model, eps=self.eps, alpha=1/255, steps=7)
        x_adv = atk(x,labels)
        noise = x_adv - x

        return x_adv, noise
    
    # Get Diff Theta

    def diff_theta(self,model,x_adv,labels):
        model.eval().cuda()

        # # initialize x_adv:
        # x_adv = x.clone()
        # x_adv += (2.0 * torch.rand(x_adv.shape).cuda() - 1.0) * self.eps # random initialize
        # x_adv = torch.clamp(x_adv, 0, 1) # clamp to RGB range [0,1]
        # x_adv = Variable(x_adv.cuda(), requires_grad=True)
        x_adv = Variable(x_adv.cuda(), requires_grad=True)
        logits_adv = model(x_adv)
        loss_adv = self.loss_fn(logits_adv, labels)
        model.zero_grad()
        loss_adv.backward()

        grad_theta = {k: copy.deepcopy(-self.lr*v.grad) for k, v in model.named_parameters()}
        grad_theta = tuple(grad_theta.values())
        return grad_theta

    """
     in_put is a combination of parameters and input samples: tuple
     where idx 0 is x
     idx 1-N is the parameters

    """
    def forward(self,t,in_put):
        # print(len(in_put))
        x= in_put[0]
        theta = in_put[1:]

        # update_model
        theta_dict = {k: v for k, v in zip(self.param_keys,theta)}
        self.model.load_state_dict(theta_dict,strict=False)


        x_adv, noise = self.diff_x(self.model, x, labels=self.labels)
        difftheta = self.diff_theta(self.model, x, labels=self.labels)

        return tuple([noise]) + difftheta
    
    def update_labels(self,labels):
        self.labels = labels


class PGD():
    def __init__(self, eps, steps=7, alpha=None, loss_fn=None, targeted=False, use_FiLM=False):
        '''
        Args:
            eps: float. noise bound.
            steps: int. PGD attack step number.
            alpha: float. step size for PGD attack.
            loss_fn: loss function which is maximized to generate adversarial images.
            targeted: bool. If Ture, do targeted attack.
        '''
        self.steps = steps
        self.eps = eps
        self.alpha = alpha if alpha else min(eps * 1.25, eps + 4/255) / steps 
        self.targeted = targeted
        self.loss_fn = loss_fn if loss_fn else nn.CrossEntropyLoss(reduction="sum")    
        self.use_FiLM = use_FiLM   


    def attack(self, model, x, labels=None, targets=None, _lambda=None, idx2BN=None):
        '''
        Args:
            x: Tensor. Original images. size=(N,C,W,H)
            model: nn.Module. The model to be attacked.
            labels: Tensor. ground truth labels for x. size=(N,). Useful only under untargeted attack.
            targets: Tensor. target attack class for x. size=(N,). Useful only under targeted attack.

        Return:
            x_adv: Tensor. Adversarial images. size=(N,C,W,H)
        '''
        # 
        model.eval().cuda()

        # initialize x_adv:
        x_adv = x.clone()
        x_adv += (2.0 * torch.rand(x_adv.shape).cuda() - 1.0) * self.eps # random initialize
        x_adv = torch.clamp(x_adv, 0, 1) # clamp to RGB range [0,1]
        x_adv = Variable(x_adv.cuda(), requires_grad=True)

        for t in range(self.steps):
            if self.use_FiLM:
                logits_adv = model(x_adv, _lambda=_lambda, idx2BN=idx2BN)
            else:
                logits_adv = model(x_adv)
            if self.targeted:
                loss_adv = - self.loss_fn(logits_adv, targets)
            else: # untargeted attack
                loss_adv = self.loss_fn(logits_adv, labels)
            grad_adv = torch.autograd.grad(loss_adv, x_adv, only_inputs=True)[0]
            x_adv.data.add_(self.alpha * torch.sign(grad_adv.data)) # gradient assend by Sign-SGD
            x_adv = linf_clamp(x_adv, _min=x-self.eps, _max=x+self.eps) # clamp to linf ball centered by x
            x_adv = torch.clamp(x_adv, 0, 1) # clamp to RGB range [0,1]
            
        return x_adv