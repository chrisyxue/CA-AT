import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

import copy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.attacks import create_attack
from core.attacks import CWLoss
from core.metrics import accuracy
from core.models import create_model

from core.utils import ctx_noparamgrad_and_eval
from core.utils import Trainer
from core.utils import set_bn_momentum
from core.utils import seed

from grad_proj import *
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WATrainer(Trainer):
    """
    Helper class for training a deep neural network with model weight averaging (identical to Gowal et al, 2020).
    Arguments:
        info (dict): dataset information.
        args (dict): input arguments.
    """
    def __init__(self, info, args):
        super(WATrainer, self).__init__(info, args)
        
        seed(args.seed)
        self.wa_model = copy.deepcopy(self.model)
        self.eval_attack = create_attack(self.wa_model, CWLoss, args.attack, args.attack_eps, 4*args.attack_iter, 
                                         args.attack_step)
        num_samples = 50000 if 'cifar' in self.params.data else 73257
        num_samples = 100000 if 'tiny-imagenet' in self.params.data else num_samples
        self.update_steps = int(np.floor(num_samples/self.params.batch_size) + 1)
        self.warmup_steps = 0.025 * self.params.num_adv_epochs * self.update_steps
    
    
    def init_optimizer(self, num_epochs):
        """
        Initialize optimizer and schedulers.
        """
        def group_weight(model):
            group_decay = []
            group_no_decay = []
            for n, p in model.named_parameters():
                if 'batchnorm' in n:
                    group_no_decay.append(p)
                else:
                    group_decay.append(p)
            assert len(list(model.parameters())) == len(group_decay) + len(group_no_decay)
            groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
            return groups
        
        self.optimizer = torch.optim.SGD(group_weight(self.model), lr=self.params.lr, weight_decay=self.params.weight_decay, 
                                         momentum=0.9, nesterov=self.params.nesterov)
        if num_epochs <= 0:
            return
        self.init_scheduler(num_epochs)
    
    
    def train(self, dataloader, epoch=0, adversarial=False, verbose=True):
        """
        Run one epoch of training.
        """
        metrics = pd.DataFrame()
        self.model.train()
        update_iter = 0
        # for data in tqdm(dataloader, desc='Epoch {}: '.format(epoch), disable=not verbose):
        for data in dataloader:
            global_step = (epoch - 1) * self.update_steps + update_iter
            if global_step == 0:
                # make BN running mean and variance init same as Haiku
                set_bn_momentum(self.model, momentum=1.0)
            elif global_step == 1:
                set_bn_momentum(self.model, momentum=0.01)
            update_iter += 1
            
            x, y = data
            x, y = x.to(device), y.to(device)
            
            clean_loss, clean_batch_metrics = self.standard_loss(x, y)
            if adversarial:
                if self.params.adver_loss == 'mart':
                    adv_loss, adv_batch_metrics = self.mart_loss(x, y)    
                elif self.params.adver_loss == 'trades':
                    adv_loss, adv_batch_metrics = self.trades_loss(x, y)
                elif self.params.adver_loss == 'adver':
                    adv_loss, adv_batch_metrics = self.adversarial_loss(x, y) 
                elif self.params.adver_loss == 'clp':
                    adv_loss, adv_batch_metrics = self.clp_loss(x,y)
                else:
                    raise ValueError('No such adver loss named '+ self.params.adver_loss)
                clean_batch_metrics.update(adv_batch_metrics)
                batch_metrics = clean_batch_metrics
            else:
                batch_metrics = clean_batch_metrics
            
            # print('clean loss',clean_loss)
            # Update Thres
            self.params = thres_update(args=self.params,epoch=epoch,clean_metrics=clean_batch_metrics,adv_metrics=adv_batch_metrics)
            gradient_adv_train(args=self.params, model=self.model, clean_loss=clean_loss, adv_loss=adv_loss, optimizer=self.optimizer)
            
            if self.params.scheduler in ['cyclic']:
                self.scheduler.step()
            
            global_step = (epoch - 1) * self.update_steps + update_iter
            ema_update(self.wa_model, self.model, global_step, decay_rate=self.params.tau, 
                       warmup_steps=self.warmup_steps, dynamic_decay=True)
            metrics = metrics.append(pd.DataFrame(batch_metrics, index=[0]), ignore_index=True)
        
        if self.params.scheduler in ['step', 'converge', 'cosine', 'cosinew']:
            self.scheduler.step()
        
        update_bn(self.wa_model, self.model) 

        if self.params.grad_track == True:
            if epoch%20 == 0 or epoch==1:
                track_grad, grad_track_layer = self.track_grad(dataloader, epoch=epoch, verbose=True)
                track_grad.to_csv(os.path.join(self.params.log_dir,str(epoch)+'_grad_track.csv'))
                grad_track_layer.to_csv(os.path.join(self.params.log_dir,str(epoch)+'_grad_track_layers.csv'))
        return dict(metrics.mean()) 

    
    def eval(self, dataloader, adversarial=False):
        """
        Evaluate performance of the model.
        """
        acc = 0.0
        self.wa_model = self.model
        self.wa_model.eval()
        
        
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if adversarial:
                with ctx_noparamgrad_and_eval(self.wa_model):
                    x_adv, _ = self.eval_attack.perturb(x, y)            
                out = self.wa_model(x_adv)
            else:
                out = self.wa_model(x)
            acc += accuracy(y, out)
        acc /= len(dataloader)
        return acc


    def save_model(self, path):
        """
        Save model weights.
        """
        torch.save({
            'model_state_dict': self.wa_model.state_dict(), 
            'unaveraged_model_state_dict': self.model.state_dict()
        }, path)

    
    def load_model(self, path):
        """
        Load model weights.
        """
        checkpoint = torch.load(path)
        if 'model_state_dict' not in checkpoint:
            raise RuntimeError('Model weights not found at {}.'.format(path))
        self.wa_model.load_state_dict(checkpoint['model_state_dict'])
    

def ema_update(wa_model, model, global_step, decay_rate=0.995, warmup_steps=0, dynamic_decay=True):
    """
    Exponential model weight averaging update.
    """
    factor = int(global_step >= warmup_steps)
    if dynamic_decay:
        delta = global_step - warmup_steps
        decay = min(decay_rate, (1. + delta) / (10. + delta)) if 10. + delta != 0 else decay_rate
    else:
        decay = decay_rate
    decay *= factor
    
    for p_swa, p_model in zip(wa_model.parameters(), model.parameters()):
        p_swa.data *= decay
        p_swa.data += p_model.data * (1 - decay)


@torch.no_grad()
def update_bn(avg_model, model):
    """
    Update batch normalization layers.
    """
    avg_model.eval()
    model.eval()
    for module1, module2 in zip(avg_model.modules(), model.modules()):
        if isinstance(module1, torch.nn.modules.batchnorm._BatchNorm):
            module1.running_mean = module2.running_mean
            module1.running_var = module2.running_var
            module1.num_batches_tracked = module2.num_batches_tracked
