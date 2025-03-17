import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.attacks import create_attack
from core.attacks.multiattack import MultiAttack

from core.metrics import accuracy
from core.models import create_model
import core.attacks.torchattackslib.torchattacks as torchattacks

from .context import ctx_noparamgrad_and_eval
from .utils import seed,get_pretrained,print_trainable_parameters

from .mart import mart_loss
from .rst import CosineLR
from .trades import trades_loss
from grad_proj import *
from .alp import clp_loss
from .label_smooth import LabelSmoothingCrossEntropyLoss, DLRLoss

# from core.utils.utils import get_pretrained

import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SCHEDULERS = ['cyclic', 'step', 'cosine', 'cosinew']


class Trainer(object):
    """
    Helper class for training a deep neural network.
    Arguments:
        info (dict): dataset information.
        args (dict): input arguments.
    """
    def __init__(self, info, args):
        super(Trainer, self).__init__()
        
        seed(args.seed)
        self.params = args
        
        # Init Standard Loss
        if self.params.use_ls:
            self.criterion = LabelSmoothingCrossEntropyLoss(smoothing=args.ls_factor)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Init Adversarial Loss
        if self.params.adver_criterion == 'ce':
            self.adv_criterion = nn.CrossEntropyLoss()
        elif self.params.adver_criterion == 'ce_ls':
            self.adv_criterion = LabelSmoothingCrossEntropyLoss(smoothing=args.ls_factor)
        elif self.params.adver_criterion == 'dlr':
            self.adv_criterion = DLRLoss(use_softmax=False)
        elif self.params.adver_criterion == 'dlr_softmax':
            self.adv_criterion = DLRLoss(use_softmax=False)
        else:
            raise ValueError('No Adv Criterion Named as: ' + str(self.params.adver_criterion))

        # Init Model
        # if self.params.use_adver_peft:
        #     # self.model = get_peft(self.params,self.model)
        #     if self.params.adver_peft_model == 'ViT-B':
        #         # TODO: Make it Formals
        #         import transformers
        #         import accelerate
        #         import peft
        #         from transformers import ViTImageProcessor, ViTForImageClassification
        #         model_checkpoint = "google/vit-base-patch16-224-in21k" 
        #         from peft import LoraConfig, get_peft_model
        #         model = ViTForImageClassification.from_pretrained(model_checkpoint)
        #         config = LoraConfig(
        #             r=16,
        #             lora_alpha=16,
        #             target_modules=["query", "value"],
        #             lora_dropout=0.1,
        #             bias="none",
        #             modules_to_save=["classifier"],
        #         )
        #         model = get_peft_model(model, config)
        #         model = torch.nn.DataParallel(model)
        #         model = model.to(device)
        #         print_trainable_parameters(model)
        #         self.model = model
        #         # pdb.set_trace()
        # else:
        self.model = create_model(args.model, args.normalize, info, device, args)

        # Init Optimizer
        self.init_optimizer(self.params.num_adv_epochs)
        
        # Init Attack for Training
        self.attack, self.eval_attack = self.init_attack(self.model, self.adv_criterion, self.params.attack, self.params.attack_eps, 
                                                         self.params.attack_iter, self.params.attack_step)
        
    
    @staticmethod
    def init_attack(model, criterion, attack_type, attack_eps, attack_iter, attack_step):
        """
        Initialize adversary.
        """
        attack = create_attack(model, criterion, attack_type, attack_eps, attack_iter, attack_step, rand_init_type='uniform')
        if attack_type in ['linf-pgd', 'l2-pgd']:
            eval_attack = create_attack(model, criterion, attack_type, attack_eps, 2*attack_iter, attack_step)
        elif attack_type in ['fgsm', 'linf-df']:
            eval_attack = create_attack(model, criterion, 'linf-pgd', 8/255, 20, 2/255)
        elif attack_type in ['fgm', 'l2-df']:
            eval_attack = create_attack(model, criterion, 'l2-pgd', 128/255, 20, 15/255)
        elif attack_type in ['fab']:
            eval_attack = create_attack(model, criterion, 'fab', 128/255, 10, 15/255)
        return attack,  eval_attack
    
    
    def init_optimizer(self, num_epochs):
        """
        Initialize optimizer and scheduler.
        """
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay, 
                                         momentum=0.9, nesterov=self.params.nesterov)
        if num_epochs <= 0:
            return
        self.init_scheduler(num_epochs)
    
        
    def init_scheduler(self, num_epochs):
        """
        Initialize scheduler.
        """
        if self.params.scheduler == 'cyclic':
            num_samples = 50000 if 'cifar10' in self.params.data else 73257
            num_samples = 100000 if 'tiny-imagenet' in self.params.data else num_samples
            update_steps = int(np.floor(num_samples/self.params.batch_size) + 1)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.params.lr, pct_start=0.25,
                                                                 steps_per_epoch=update_steps, epochs=int(num_epochs))
        elif self.params.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=0.1, milestones=[100, 105])    
        elif self.params.scheduler == 'cosine':
            self.scheduler = CosineLR(self.optimizer, max_lr=self.params.lr, epochs=int(num_epochs))
        elif self.params.scheduler == 'cosinew':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.params.lr, pct_start=0.025, 
                                                                 total_steps=int(num_epochs))
        else:
            self.scheduler = None
    
    def track_grad_multiattack(self, dataloader, epoch=0, verbose=True):
        """
        Track the gradients of a list of adversarial attack 
        """
        multiattack = MultiAttack(model=self.model)
        track_grad = {}
        count = 0
        for data in tqdm(dataloader, desc='Check Epoch {}: '.format(epoch), disable=not verbose):
            x, y = data
            x, y = x.to(device), y.to(device)

            # multiattack
            with ctx_noparamgrad_and_eval(self.model):
                x_adv_lst, attk_name_lst = multiattack.perturb(x, y)
            
            for idx in range(len(x_adv_lst)):
                clean_loss, clean_batch_metrics = self.standard_loss(x, y)
                x_adv = x_adv_lst[idx]
                attk_name = attk_name_lst[idx]

                self.optimizer.zero_grad()
                out = self.model(x_adv)
                adv_loss = self.criterion(out, y)
                preds = out.detach()
                adv_batch_metrics = {'adv_loss': adv_loss.item()}
                adv_batch_metrics.update({'adv_acc': accuracy(y, preds)}) 

                _, _, clean_gradients, _, _, adv_gradients = get_gradients_per_batch(self.model, clean_loss, adv_loss)            
                if count == 0:
                    track_grad[attk_name] = pd.DataFrame([track_gradient(clean_gradients,adv_gradients)])
                else:
                    track_grad[attk_name] += pd.DataFrame([track_gradient(clean_gradients,adv_gradients)])
                # print(attk_name)

            count = count + 1
            # pdb.set_trace()
            for attk_name in attk_name_lst:
                track_grad[attk_name] = track_grad[attk_name]/count

        return track_grad,attk_name_lst


    def track_grad(self, dataloader, epoch=0, verbose=True):
        track_grad = {}
        count = 0
        for data in tqdm(dataloader, desc='Check Epoch {}: '.format(epoch), disable=not verbose):
            x, y = data
            x, y = x.to(device), y.to(device)
            clean_loss, clean_batch_metrics = self.standard_loss(x, y)
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
            batch_metrics = clean_batch_metrics.update(adv_batch_metrics)

            clean_layer_names, clean_layer_grad_dims,clean_gradients,adv_layer_names, adv_layer_grad_dims,adv_gradients = get_gradients_per_batch(self.model, clean_loss, adv_loss)
            if count == 0:
                track_grad = pd.DataFrame([track_gradient(clean_gradients,adv_gradients)])
                grad_track_layer  = track_gradient_per_layer(clean_layer_names,clean_layer_grad_dims,clean_gradients,adv_layer_names,adv_layer_grad_dims,adv_gradients)
            else:
                track_grad += pd.DataFrame([track_gradient(clean_gradients,adv_gradients)])
                grad_track_layer += track_gradient_per_layer(clean_layer_names,clean_layer_grad_dims,clean_gradients,adv_layer_names,adv_layer_grad_dims,adv_gradients)
            count = count + 1
        
        track_grad = track_grad/count
        grad_track_layer = grad_track_layer/count
        return track_grad, grad_track_layer

    def train(self, dataloader, epoch=0, adversarial=False, verbose=True):
        """
        Run one epoch of training.
        """
        metrics = pd.DataFrame()
        self.model.train()

        # for data in tqdm(dataloader, desc='Epoch {}: '.format(epoch), disable=not verbose):
        for data in dataloader:
            x, y = data
            x, y = x.to(device), y.to(device)
            
            clean_loss, clean_batch_metrics = self.standard_loss(x, y)
            
            # print(clean_loss)

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
            
            gradient_adv_train(args=self.params, model=self.model, clean_loss=clean_loss, adv_loss=adv_loss, optimizer=self.optimizer)
            if self.params.scheduler in ['cyclic']:
                self.scheduler.step()
            
            metrics = pd.concat([metrics,pd.DataFrame(batch_metrics, index=[0])],ignore_index=True)
        
        if self.params.scheduler in ['step', 'converge', 'cosine', 'cosinew']:
            self.scheduler.step()
        
        if self.params.grad_track == True:
            if epoch%20 == 0 or epoch==1:
                track_grad, grad_track_layer = self.track_grad(dataloader, epoch=epoch, verbose=True)
                track_grad.to_csv(os.path.join(self.params.log_dir,str(epoch)+'_grad_track.csv'))
                grad_track_layer.to_csv(os.path.join(self.params.log_dir,str(epoch)+'_grad_track_layers.csv'))
        
        if self.params.grad_track_multiattack == True:
            if epoch%20 == 0 or epoch==1:
                track_grad_multiattack,attk_name_lst = self.track_grad_multiattack(dataloader, epoch=epoch, verbose=True)
                for attk_name in attk_name_lst:
                    track_grad_multiattack[attk_name].to_csv(os.path.join(self.params.log_dir,str(attk_name)+str(epoch)+'_multi_grad_track.csv'))
        return dict(metrics.mean())
    
    
    def standard_loss(self, x, y):
        """
        Standard training.
        """
        self.optimizer.zero_grad()
        out = self.model(x)
        loss = self.criterion(out, y)
        
        preds = out.detach()
        batch_metrics = {'clean_loss': loss.item(), 'clean_acc': accuracy(y, preds)}
        return loss, batch_metrics
    
    
    def adversarial_loss(self, x, y):
        """
        Adversarial training (Madry et al, 2017).
        """
        with ctx_noparamgrad_and_eval(self.model):
            x_adv, _ = self.attack.perturb(x, y)
        
        self.optimizer.zero_grad()

        y_adv = y
        out = self.model(x_adv)
        loss = self.criterion(out, y_adv)
        preds = out.detach()
        batch_metrics = {'adv_loss': loss.item()}
        batch_metrics.update({'adv_acc': accuracy(y, preds)}) 
        return loss, batch_metrics
    
    
    def trades_loss(self, x, y):
        """
        TRADES training.
        """
        loss, batch_metrics = trades_loss(self.model, x, y, self.optimizer, step_size=self.params.attack_step, 
                                          epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter,attack=self.params.attack)
        return loss, batch_metrics  

    
    def mart_loss(self, x, y):
        """
        MART training.
        """
        loss, batch_metrics = mart_loss(self.model, x, y, self.optimizer, step_size=self.params.attack_step, 
                                        epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter, 
                                        attack=self.params.attack)
        return loss, batch_metrics  
    
    def clp_loss(self,x,y):
        """
        CLP training
        """
        loss, batch_metrics = clp_loss(self.model, x, y, self.optimizer, step_size=self.params.attack_step, 
                                        epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter,attack=self.params.attack)
        return loss, batch_metrics  
    
    def eval(self, dataloader, adversarial=False):
        """
        Evaluate performance of the model.
        """
        acc = 0.0
        correct_num = 0
        self.model.eval()
        
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if adversarial:
                with ctx_noparamgrad_and_eval(self.model):
                    x_adv, _ = self.eval_attack.perturb(x, y)            
                out = self.model(x_adv)
            else:
                out = self.model(x)
            acc += accuracy(y, out)
            correct_num += accuracy(y, out)*x.shape[0]
            # print(acc)
            # print(x.shape[0])
        # pdb.set_trace()
        acc /= len(dataloader)
        return acc
    
    
    def save_model(self, path):
        """
        Save model weights.
        """
        torch.save({'model_state_dict': self.model.state_dict()}, path)

    
    def load_model(self, path, load_opt=True):
        """
        Load model weights.
        """
        checkpoint = torch.load(path)
        if 'model_state_dict' not in checkpoint:
            raise RuntimeError('Model weights not found at {}.'.format(path))
        self.model.load_state_dict(checkpoint['model_state_dict'])
