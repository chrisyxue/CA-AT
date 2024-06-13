'''
PGDAT

Generate Adver samples in different epoches in adversarial training

(Also store models for different epoches)
'''
from torchvision import transforms
from torch.utils.data import DataLoader

import os, argparse, time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam, lr_scheduler

from models.cifar10.resnet import ResNet34, ResNet34_ENC, ResNet_CLS
from models.svhn.wide_resnet import WRN16_8
from models.stl10.wide_resnet import WRN40_2
from models.model_combine import ComModel
from models.cifar10.simple_cnn import SimpleCNN_ENC, SimpleCNN_CLS

from dataloaders.cifar10 import cifar10_dataloaders
from dataloaders.svhn import svhn_dataloaders
from dataloaders.stl10 import stl10_dataloaders

from utils.utils import *
from utils.context import ctx_noparamgrad_and_eval
from attacks.pgd import PGD

from torchvision.utils import save_image
from torch import nn

from avalanche.training.utils import copy_params_dict, zerolike_params_dict
from avalanche.benchmarks.classic import SplitCIFAR10, SplitCIFAR100, SplitCIFAR110, SplitMNIST, SplitTinyImageNet
from collections import defaultdict


from avalanche.models import SimpleCNN
import pandas as pd

from Continual.Finetune import Finetune
from Continual.AGEM import AGEM, update_memory

import copy
import pdb

parser = argparse.ArgumentParser(description='CIFAR10 Training against DDN Attack')
parser.add_argument('--gpu', default='0')
parser.add_argument('--cpus', default=4)
# dataset:
parser.add_argument('--dataset', '--ds', default='cifar100', choices=['cifar10', 'svhn', 'stl10','cifar100'], help='which dataset to use')
# optimization parameters:
parser.add_argument('--batch_size', '-b', default=256, type=int, help='mini-batch size')
parser.add_argument('--epochs', '-e', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--decay_epochs', '--de', default=[50,150], nargs='+', type=int, help='milestones for multisteps lr decay')
parser.add_argument('--opt', default='sgd', choices=['sgd', 'adam'], help='which optimizer to use')
parser.add_argument('--decay', default='cos', choices=['cos', 'multisteps'], help='which lr decay method to use')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
# adv parameters:
parser.add_argument('--targeted', action='store_true', help='If true, targeted attack')
parser.add_argument('--eps', type=int, default=8)
parser.add_argument('--steps', type=int, default=7)
# loss parameters:
parser.add_argument('--Lambda', default=0.5, type=float, help='adv loss tradeoff parameter')

# continual learning
parser.add_argument('--model', default='ResNet34', type=str, choices=['ResNet34','SimpleCNN']) # the number of tasks
parser.add_argument('--task_num', default=5, type=int, help='adv loss tradeoff parameter') # the number of tasks
parser.add_argument('--from_scratch', action="store_true") # the number of tasks
parser.add_argument('--method', default='AGEM',choices=['Finetune','AGEM']) # rehersal methods

# For Agem
parser.add_argument('--patterns_per_experience', default=400, type=int)
parser.add_argument('--sample_size', default=100, type=int)
parser.add_argument('--agem_mode', default='avg_before', type=str, choices=['avg_before','avg_after'])

# others:
parser.add_argument('--resume', action='store_true', help='If true, resume from early stopped ckpt')
args = parser.parse_args()
print(args)

if args.method == 'AGEM':
    buffers = []
    buffer_dliter = None

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.backends.cudnn.benchmark = True

if args.method == 'EWC':
    saved_params = defaultdict(list)
    importances_list = defaultdict(list)
    if args.ewc_mode == 'separate':
        keep_importance_data = True
    else:
        keep_importance_data = False

def label_transform(labels,classes_in_this_experience):
    for i in range(len(classes_in_this_experience)):
        labels[labels==classes_in_this_experience[i]] = i
    return labels

def weights_init(m): 
    if isinstance(m, nn.Conv2d): 
        nn.init.xavier_normal_(m.weight.data) 
        # nn.init.xavier_normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias, 0)

def init_model(args):
    if args.dataset == 'cifar100':
        if args.model == 'ResNet34':
            enc_fn = ResNet34_ENC
            enc = enc_fn().cuda()
            cls_fn = ResNet_CLS
            classifier = cls_fn(num_classes=int(100/args.task_num)).cuda()
            model = ComModel(enc,classifier)
        elif args.model == 'SimpleCNN':
            enc = SimpleCNN_ENC().cuda()
            classifier = SimpleCNN_CLS(num_classes=int(100/args.task_num)).cuda()
            model = ComModel(enc,classifier)
    # elif args.model == 'SimpleCNN':
    #     model = SimpleCNN(num_classes=int(100/args.task_num)).cuda()
    elif args.dataset == 'cifar10':
        if args.model == 'ResNet34':
            model_fn = ResNet34
            model = model_fn(num_classes=int(10/args.task_num)).cuda()
    model = torch.nn.DataParallel(model)
    model.apply(weights_init)

    return model


# data loader (cotinual):
if args.dataset == 'cifar100':
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = SplitCIFAR100(n_experiences=args.task_num,seed=26,
                            return_task_id=True,
                            train_transform=train_transform,eval_transform=test_transform)
elif args.dataset == 'cifar10':
    dataset = SplitCIFAR10(n_experiences=args.task_num,
    seed=26,
    return_task_id=False,
    dataset_root='/localscratch2/xuezhiyu/datasets'
    )
train_stream = dataset.train_stream
test_stream = dataset.test_stream

# Model
model = init_model(args)

# mkdirs:
# model_str = model_fn.__name__
model_str = args.model
if args.opt == 'sgd':
    opt_str = 'e%d-b%d_sgd-lr%s-m%s-wd%s' % (args.epochs, args.batch_size, args.lr, args.momentum, args.wd)
elif args.opt == 'adam':
    opt_str = 'e%d-b%d_adam-lr%s-wd%s' % (args.epochs, args.batch_size, args.lr, args.wd)
if args.decay == 'cos':
    decay_str = 'cos'
elif args.decay == 'multisteps':
    decay_str = 'multisteps-%s' % args.decay_epochs
attack_str = 'targeted' if args.targeted else 'untargeted' + '-pgd-%d-%d' % (args.eps, args.steps)
loss_str = 'lambda%s' % (args.Lambda)


save_folder = os.path.join('/localscratch2/xuezhiyu/Adver_Continual_Rehearsal', args.dataset, args.method, 'Task_Num_'+str(args.task_num),model_str, '%s_%s' % (attack_str, loss_str))

if args.method == 'AGEM':
    save_folder = os.path.join('/localscratch2/xuezhiyu/Adver_Continual_Rehearsal', args.dataset, args.method, args.agem_mode, 'Task_Num_'+str(args.task_num),model_str, '%s_%s' % (attack_str, loss_str))


save_folder_models = os.path.join(save_folder,'models')
save_folder_data = os.path.join(save_folder,'adver_data')

create_dir(save_folder_models)
create_dir(save_folder_data)


# if args.decay == 'cos':
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
# elif args.decay == 'multisteps':
#     scheduler = lr_scheduler.MultiStepLR(optimizer, args.decay_epochs, gamma=0.1)

attacker = PGD(eps=args.eps/255, steps=args.steps)

start_epoch = 0
best_TA, best_ATA = 0, 0
# training curve lists:
training_loss, val_TA, val_ATA = [], [], []

# store acc/ader-acc for each task

criterion = nn.CrossEntropyLoss()

for experience in train_stream:
    print("Start of task ", experience.task_label)
    print('Classes in this task:', experience.classes_in_this_experience)

    current_training_set = experience.dataset
    print('Task {}'.format(experience.task_label))
    print('This task contains', len(current_training_set), 'training examples')
    
    print(experience.current_experience)
    current_test_set = test_stream[experience.current_experience].dataset
    print('This task contains', len(current_test_set), 'test examples')

    train_loader = DataLoader(current_training_set, batch_size=args.batch_size, shuffle=True, num_workers=3,
                                drop_last=True, pin_memory=True)
    val_loader = DataLoader(current_test_set, batch_size=args.batch_size, shuffle=False, num_workers=3,
                                drop_last=False, pin_memory=True)
    current_task = experience.current_experience
    
    # optimizer:
    if args.opt == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    elif args.opt == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # schedular
    if args.decay == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.decay == 'multisteps':
        scheduler = lr_scheduler.MultiStepLR(optimizer, args.decay_epochs, gamma=0.1)
    
    # best TA/ATA
    best_TA, best_ATA = 0, 0
    
    for epoch in range(start_epoch, args.epochs):
        fp = open(os.path.join(save_folder, 'train_log.txt'), 'a+')
        start_time = time.time()
        ## training:
        model.train()
        requires_grad_(model, True)
        accs, accs_adv, losses = AverageMeter(), AverageMeter(), AverageMeter()

        if args.method == 'AGEM':
            AGEM(train_loader,model,attacker,optimizer,args,scheduler,accs,accs_adv,losses,buffers,buffer_dliter,criterion)



        # for i, data in enumerate(train_loader):
        #     imgs, labels = data[0],data[1]
        #     # labels = label_transform(labels,experience.classes_in_this_experience)
        #     # print(labels)
        #     imgs, labels = imgs.cuda(), labels.cuda()
        #     # generate adversarial images:
        #     if args.Lambda != 0:
        #         with ctx_noparamgrad_and_eval(model):
        #             imgs_adv = attacker.attack(model, imgs, labels)
        #         logits_adv = model(imgs_adv.detach())
        #     # logits for clean imgs:
        #     logits = model(imgs)

        #     # loss and update:
        #     loss = F.cross_entropy(logits, labels)
        #     if args.Lambda != 0:
        #         loss = (1-args.Lambda) * loss + args.Lambda * F.cross_entropy(logits_adv, labels)
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

            # if save_this_epoch:
            #     store_data(imgs_adv,labels, save_folder_data_epoch, 0, data_type='train')

            # proximal gradient for channel pruning:
            # current_lr = scheduler.get_lr()[0]
            # # metrics:
            # accs.append((logits.argmax(1) == labels).float().mean().item())
            # if args.Lambda != 0:
            #     accs_adv.append((logits_adv.argmax(1) == labels).float().mean().item())
            # losses.append(loss.item())
        current_lr = scheduler.get_lr()[0]
        if args.Lambda != 0:
            if accs_adv.avg > best_ATA:
                best_ATA = accs_adv.avg
                torch.save(model.module.enc.state_dict(),os.path.join(save_folder_models,'Task_'+str(current_task)+'_enc_best.pth'))
                torch.save(model.module.cls.state_dict(),os.path.join(save_folder_models,'Task_'+str(current_task)+'_cls_best.pth'))
        else:
            if accs.avg > best_ATA:
                best_ATA = accs.avg
                torch.save(model.module.enc.state_dict(),os.path.join(save_folder_models,'Task_'+str(current_task)+'_enc_best.pth'))
                torch.save(model.module.cls.state_dict(),os.path.join(save_folder_models,'Task_'+str(current_task)+'_cls_best.pth'))

        if (epoch+1) % 10 == 0:
            model.eval()
            requires_grad_(model, False)
            train_str = 'Train Task %d | Epoch %d | Train | Loss: %.4f, SA: %.4f' % (current_task, epoch, losses.avg, accs.avg)
            if args.Lambda != 0:
                train_str += ', RA: %.4f' % (accs_adv.avg)
            print(train_str)

            val_accs, val_accs_adv = AverageMeter(), AverageMeter()
            for i, data in enumerate(val_loader):
                imgs, labels = data[0],data[1]
                # labels = label_transform(labels,experience.classes_in_this_experience)
                imgs, labels = imgs.cuda(), labels.cuda()

                # generate adversarial images:
                with ctx_noparamgrad_and_eval(model):
                    imgs_adv = attacker.attack(model, imgs, labels)
                linf_norms = (imgs_adv - imgs).view(imgs.size()[0], -1).norm(p=np.Inf, dim=1)
                logits_adv = model(imgs_adv.detach())
                # logits for clean imgs:
                logits = model(imgs)

                # if save_this_epoch:
                #     store_data(imgs_adv,labels, save_folder_data_epoch, 0, data_type='val')
                
                val_accs.append((logits.argmax(1) == labels).float().mean().item())
                val_accs_adv.append((logits_adv.argmax(1) == labels).float().mean().item())
            
            val_str = 'Val Task %d | Epoch %d | Validation | Time: %.4f | lr: %s | SA: %.4f, RA: %.4f, linf: %.4f - %.4f' % (
                current_task, epoch, (time.time()-start_time), current_lr, val_accs.avg, val_accs_adv.avg, 
                torch.min(linf_norms).data, torch.max(linf_norms).data)
            print(val_str)

        scheduler.step()

   #  eval_this_epoch = (epoch % 20 == 0) or (epoch>=int(0.7*args.epochs))

    # if eval_this_epoch:
    res_task = pd.DataFrame(columns=['Acc','Adver_Acc'])

    for experience in test_stream[:current_task+1]: 
        model.eval()
        requires_grad_(model, False)
        print('Task {}'.format(experience.task_label)) 
        print('Classes in this task:', experience.classes_in_this_experience)   
        val_accs, val_accs_adv = AverageMeter(), AverageMeter()
        task_id_val = experience.current_experience
        
        model = init_model(args)
        # shared parameters
        model.module.enc.load_state_dict(torch.load(os.path.join(save_folder_models,'Task_'+str(current_task)+'_enc_best.pth')))
        # task specific parameters
        model.module.cls.load_state_dict(torch.load(os.path.join(save_folder_models,'Task_'+str(task_id_val)+'_cls_best.pth')))
        
        current_test_set = experience.dataset


        val_loader = DataLoader(current_test_set, batch_size=args.batch_size, shuffle=False, num_workers=3,drop_last=False, pin_memory=True)

        for i, data in enumerate(val_loader):
            imgs, labels = data[0],data[1]
            labels = label_transform(labels,experience.classes_in_this_experience)
            imgs, labels = imgs.cuda(), labels.cuda()

            # generate adversarial images:
            with ctx_noparamgrad_and_eval(model):
                imgs_adv = attacker.attack(model, imgs, labels)
            linf_norms = (imgs_adv - imgs).view(imgs.size()[0], -1).norm(p=np.Inf, dim=1)
            logits_adv = model(imgs_adv.detach())
            # logits for clean imgs:
            logits = model(imgs)

            # if save_this_epoch:
            #     store_data(imgs_adv,labels, save_folder_data_epoch, 0, data_type='val')
            
            val_accs.append((logits.argmax(1) == labels).float().mean().item())
            val_accs_adv.append((logits_adv.argmax(1) == labels).float().mean().item())
        
        val_str = 'Final Val Task %d | Epoch %d | Validation | Time: %.4f | lr: %s | SA: %.4f, RA: %.4f, linf: %.4f - %.4f' % (
            task_id_val, epoch, (time.time()-start_time), current_lr, val_accs.avg, val_accs_adv.avg, 
            torch.min(linf_norms).data, torch.max(linf_norms).data)
        print(val_str)
        fp.write(val_str + '\n')

        res_task.loc[task_id_val,'Acc'] = val_accs.avg
        res_task.loc[task_id_val,'Adver_Acc'] = val_accs_adv.avg
        # print(res_task)
        res_task.to_csv(os.path.join(save_folder,'result_task_'+str(current_task)+'.csv'))
    
    if args.method == 'AGEM':
        store_dataset = train_stream[current_task].dataset
        buffers,buffer_dliter,buffer_dataloader = update_memory(store_dataset,args.patterns_per_experience,buffers,args.sample_size)
        pdb.set_trace()
    
    model = init_model(args)
    model.module.enc.load_state_dict(torch.load(os.path.join(save_folder_models,'Task_'+str(current_task)+'_enc_best.pth')))
    # model.module.cls.load_state_dict(torch.load(os.path.join(save_folder_models,'Task_'+str(current_task)+'_cls_best.pth')))
    
