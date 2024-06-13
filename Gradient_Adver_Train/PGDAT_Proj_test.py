'''
PGDAT
'''

import os, argparse, time
import numpy as np
import matplotlib.pyplot as plt
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam, lr_scheduler



from dataloaders.cifar10 import cifar10_dataloaders
from dataloaders.svhn import svhn_dataloaders
from dataloaders.stl10 import stl10_dataloaders
from dataloaders.cifar100 import cifar100_dataloaders

from utils.utils import *
from utils.context import ctx_noparamgrad_and_eval
from attacks.pgd import PGD
# from attacks.cattack import *
import pdb
import copy
from attacks.pdg_grad_proj import *
import torch.nn.functional as F 

from mpl_toolkits.mplot3d import Axes3D
from attack_test_2 import eval_adv_test_whitebox
from models.wideresnet import  wideresnet
from models.resnet import resnet

parser = argparse.ArgumentParser(description='CIFAR10 Training against DDN Attack')
parser.add_argument('--gpu', default='1',type=str)
parser.add_argument('--cpus', default=4)
# dataset:
parser.add_argument('--dataset', '--ds', default='cifar10', choices=['cifar10', 'svhn', 'stl10', 'cifar100'], help='which dataset to use')
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
# others:
parser.add_argument('--resume', action='store_true', help='If true, resume from early stopped ckpt')
parser.add_argument('--grad_norm', action='store_true')
# parser.add_argument('--grad_proj', action='store_true')
parser.add_argument('--grad_proj', default='avg',type=str,choices=['avg','thres','orth'])
parser.add_argument('--grad_thres', default=0.5,type=float)
parser.add_argument('--use_trades',action='store_true',help='use TRADES or not')
parser.add_argument('--model', default='None',choices=['resnet18','resnet34','resnet50','resnet101','WRN-16-8','WRN-40-2','WRN-28-10','None'])

# use clean data pretraining model or not
parser.add_argument('--use_pretrain',action='store_true',help='use the model pretrained on standard training')

parser.add_argument('--save_folder',required=True,type=str)

# For attack test
parser.add_argument('--step_size', type=int, default=1)
args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.backends.cudnn.benchmark = False
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False
seed_torch()

data_dir = '/root/dataset'
if args.dataset == 'cifar10':
    train_loader, val_loader, _ = cifar10_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus, data_dir=data_dir)
    num_classes = 10
elif args.dataset == 'svhn':
    train_loader, val_loader, _ = svhn_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus, data_dir=data_dir)
    num_classes = 10
elif args.dataset == 'stl10':
    train_loader, val_loader = stl10_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus, data_dir=data_dir)
    num_classes = 10
elif args.dataset == 'cifar100':
    train_loader, val_loader, _ = cifar100_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus, data_dir=data_dir)
    num_classes = 100

# model:
if args.model == 'None':
    if args.dataset == 'cifar10':
        args.model = 'resnet34' 
    elif args.dataset == 'svhn':
        args.model = 'WRN-16-8'
    elif args.dataset == 'stl10':
        args.model = 'WRN-40-2' 
    elif args.dataset == 'cifar100':
        args.model == 'resnet50'

if 'res' in args.model:
    model = resnet(args.model,num_classes=num_classes)
elif 'WRN' in args.model:
    model = wideresnet(args.model,num_classes=num_classes)
else:
    raise ValueError('No model '+str(args.model))

model = model.cuda()

# mkdirs:
model_str = args.model
if args.opt == 'sgd':
    opt_str = 'e%d-b%d_sgd-lr%s-m%s-wd%s' % (args.epochs, args.batch_size, args.lr, args.momentum, args.wd)
elif args.opt == 'adam':
    opt_str = 'e%d-b%d_adam-lr%s-wd%s' % (args.epochs, args.batch_size, args.lr, args.wd)
if args.decay == 'cos':
    decay_str = 'cos'
elif args.decay == 'multisteps':
    decay_str = 'multisteps-%s' % args.decay_epochs

"""
save_folder should be loaded in parameters
"""
save_folder = args.save_folder
# # attack_str = 'targeted' if args.targeted else 'untargeted' + '-pgd-grad-%d-%d' % (args.eps, args.steps)
# attack_str = 'targeted' if args.targeted else 'untargeted' + '-pgd-%d-%d' % (args.eps, args.steps)
# # if args.grad_proj:
# #     attack_str += '-proj'
# attack_str += '-proj_'+str(args.grad_proj)
# if args.grad_proj == 'thres':
#     attack_str += str(args.grad_thres)
    
# if args.use_trades is True:
#     attack_str += '-TRADES'

# loss_str = 'lambda%s' % (args.Lambda)
# save_folder = os.path.join('/localscratch2/xuezhiyu/Gradient_Adver_Train', args.dataset, model_str, '%s_%s_%s_%s' % (attack_str, opt_str, decay_str, loss_str))


# create_dir(save_folder)

# optimizer:
if args.opt == 'sgd':
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
elif args.opt == 'adam':
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
if args.decay == 'cos':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
elif args.decay == 'multisteps':
    scheduler = lr_scheduler.MultiStepLR(optimizer, args.decay_epochs, gamma=0.1)

# attacker:
attacker = PGD(eps=args.eps/255, steps=args.steps)


# load ckpt:
if os.path.exists(save_folder):
    last_epoch, best_TA, best_ATA, training_loss, val_TA, val_ATA \
         = load_ckpt(model, optimizer, scheduler, os.path.join(save_folder, 'latest.pth'))
    start_epoch = last_epoch + 1
else:
    raise ValueError('No file '+ save_folder)

# Test Adv Acc on Different Attacks
test_res = eval_adv_test_whitebox(model, val_loader, args)
test_name = 'Step_'+str(args.steps)+'_StepSize_'+str(args.step_size)+'_test_res.csv'
test_res.to_csv(os.path.join(save_folder,test_name))
# pdb.set_trace()

# val_accs, val_accs_adv = AverageMeter(), AverageMeter()
# clean_grad_norm_lst,adv_grad_norm_lst,cos_grad_lst = [],[],[]
# criterion = torch.nn.CrossEntropyLoss()
# fp = open(os.path.join(save_folder, 'test_log.txt'), 'a+')
# start_time = time.time()
# for i, (imgs, labels) in enumerate(val_loader):
#     imgs, labels = imgs.cuda(), labels.cuda()
    
#     # generate adversarial images:
#     with ctx_noparamgrad_and_eval(model):
#         imgs_adv = attacker.attack(model, imgs, labels)
#     linf_norms = (imgs_adv - imgs).view(imgs.size()[0], -1).norm(p=np.Inf, dim=1)
#     logits_adv = model(imgs_adv.detach())
    
#     # logits for clean imgs:
#     logits = model(imgs)

#     val_accs.append((logits.argmax(1) == labels).float().mean().item())
#     val_accs_adv.append((logits_adv.argmax(1) == labels).float().mean().item())

     
#     clean_gradients,adv_gradients = get_gradients_per_batch(args, model, criterion, optimizer, imgs, imgs_adv, labels)
#     # import pdb;pdb.set_trace()
#     cos_grad = F.cosine_similarity(clean_gradients,adv_gradients,dim=-1)
#     clean_l2 = torch.norm(clean_gradients)
#     adv_l2 = torch.norm(adv_gradients)


#     cos_grad_lst.append(cos_grad.detach().cpu().item())
#     clean_grad_norm_lst.append(clean_l2.detach().cpu().item())
#     adv_grad_norm_lst.append(adv_l2.detach().cpu().item())


# # cos_grad_lst = np.concatenate(cos_grad_lst)
# # clean_grad_norm_lst = np.concatenate(clean_grad_norm_lst)
# # adv_grad_norm_lst = np.concatenate(adv_grad_norm_lst)
# val_str = 'Validation | Time: %.4f | SA: %.4f, RA: %.4f, linf: %.4f - %.4f' % (
#     (time.time()-start_time), val_accs.avg, val_accs_adv.avg, 
#     torch.min(linf_norms).data, torch.max(linf_norms).data)
# print(val_str)
# fp.write(val_str + '\n')


# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(val_accs.values,val_accs_adv.values,cos_grad_lst)
# plt.title(val_str)
# plt.savefig(os.path.join(save_folder, '3d_vis.png'))

# plt.figure()
# plt.scatter(val_accs.values,cos_grad_lst)
# plt.title(val_str)
# plt.savefig(os.path.join(save_folder, 'acc-cos.png'))

# plt.figure()
# plt.scatter(val_accs_adv.values,cos_grad_lst)
# plt.title(val_str)
# plt.savefig(os.path.join(save_folder, 'advacc-cos.png'))

# from sklearn.linear_model import LinearRegression
# lr = LinearRegression()
# X = np.array([val_accs.values,val_accs_adv.values]).T
# y = np.array(cos_grad_lst)
# lr.fit(X, y)
# y_pre = lr.predict(X)
# lr_str = 'Acc_'+str(round(lr.coef_[0],4)) + '_AdvAcc_'+str(round(lr.coef_[-1],4)) + '_Score_' + str(lr.score(X, y))

# plt.figure()
# plt.scatter(y_pre,y)
# plt.title(lr_str)
# plt.savefig(os.path.join(save_folder, 'lr_acc_cos.png'))


# plt.figure()
# acc_dif = [val_accs.values[i] - val_accs_adv.values[i] for i in range(int(len(val_accs_adv.values)))]
# plt.scatter(acc_dif,cos_grad_lst)
# plt.title(val_str)
# plt.savefig(os.path.join(save_folder, 'accdif-cos.png'))


# plt.figure()
# plt.hist(cos_grad_lst)
# plt.title(val_str)
# plt.savefig(os.path.join(save_folder, 'cos_sim.png'))

# plt.figure()
# plt.hist(clean_grad_norm_lst)
# plt.title(val_str)
# plt.savefig(os.path.join(save_folder, 'clean_norm.png'))

# plt.figure()
# plt.hist(adv_grad_norm_lst)
# plt.title(val_str)
# plt.savefig(os.path.join(save_folder, 'adv_norm.png'))

# import pdb;pdb.set_trace()

# ## training:
# for epoch in range(start_epoch, args.epochs):
#     fp = open(os.path.join(save_folder, 'train_log.txt'), 'a+')
#     start_time = time.time()
#     ## training:
#     model.train()
#     requires_grad_(model, True)
#     accs, accs_adv, losses = AverageMeter(), AverageMeter(), AverageMeter()
#     for i, (imgs, labels) in enumerate(train_loader):
#         imgs, labels = imgs.cuda(), labels.cuda()

#         # generate adversarial images:
#         if args.Lambda != 0:
#             with ctx_noparamgrad_and_eval(model):
#                 imgs_adv = attacker.attack(model, imgs, labels)
        
#         model,optimizer = gradient_adv_train(args,model, criterion, optimizer, imgs, imgs_adv, labels, epoch)
#         logits = model(imgs)
#         logits_adv = model(imgs_adv.detach())
        
#         loss = F.cross_entropy(logits, labels)
#         # proximal gradient for channel pruning:
#         current_lr = scheduler.get_lr()[0]

#         # metrics:
#         accs.append((logits.argmax(1) == labels).float().mean().item())
#         if args.Lambda != 0:
#             accs_adv.append((logits_adv.argmax(1) == labels).float().mean().item())
#         losses.append(loss.item())

#         if i % 50 == 0:
#             train_str = 'Epoch %d-%d | Train | Loss: %.4f, SA: %.4f' % (epoch, i, losses.avg, accs.avg)
#             if args.Lambda != 0:
#                 train_str += ', RA: %.4f' % (accs_adv.avg)
#             # if np.isnan(losses.avg):
#             #     pdb.set_trace()
#             # if torch.isnan(losses.avg):
#             #     pdb.set_trace()
#             print(train_str)
#     # lr schedualr update at the end of each epoch:
#     scheduler.step()


#     ## validation:
#     model.eval()
#     requires_grad_(model, False)
#     print(model.training)

#     if args.dataset == 'cifar10':
#         eval_this_epoch = (epoch % 10 == 0) or (epoch>=int(0.7*args.epochs)) # boolean
#     elif args.dataset == 'svhn':
#         eval_this_epoch = (epoch % 10 == 0) or (epoch>=int(0.25*args.epochs)) # boolean
    
#     if eval_this_epoch:
#         val_accs, val_accs_adv = AverageMeter(), AverageMeter()
#         for i, (imgs, labels) in enumerate(val_loader):
#             imgs, labels = imgs.cuda(), labels.cuda()

#             # generate adversarial images:
#             with ctx_noparamgrad_and_eval(model):
#                 imgs_adv = attacker.attack(model, imgs, labels)
#             linf_norms = (imgs_adv - imgs).view(imgs.size()[0], -1).norm(p=np.Inf, dim=1)
#             logits_adv = model(imgs_adv.detach())
#             # logits for clean imgs:
#             logits = model(imgs)

#             val_accs.append((logits.argmax(1) == labels).float().mean().item())
#             val_accs_adv.append((logits_adv.argmax(1) == labels).float().mean().item())

#         val_str = 'Epoch %d | Validation | Time: %.4f | lr: %s | SA: %.4f, RA: %.4f, linf: %.4f - %.4f' % (
#             epoch, (time.time()-start_time), current_lr, val_accs.avg, val_accs_adv.avg, 
#             torch.min(linf_norms).data, torch.max(linf_norms).data)
#         print(val_str)
#         fp.write(val_str + '\n')

#     # save loss curve:
#     training_loss.append(losses.avg)
#     plt.plot(training_loss)
#     plt.grid(True)
#     plt.savefig(os.path.join(save_folder, 'training_loss.png'))
#     plt.close()

#     if eval_this_epoch:
#         val_TA.append(val_accs.avg) 
#         plt.plot(val_TA, 'r')
#         val_ATA.append(val_accs_adv.avg)
#         plt.plot(val_ATA, 'g')
#         plt.grid(True)
#         plt.savefig(os.path.join(save_folder, 'val_acc.png'))
#         plt.close()
#     else:
#         val_TA.append(val_TA[-1]) 
#         plt.plot(val_TA, 'r')
#         val_ATA.append(val_ATA[-1])
#         plt.plot(val_ATA, 'g')
#         plt.grid(True)
#         plt.savefig(os.path.join(save_folder, 'val_acc.png'))
#         plt.close()

#     # save pth:
#     if eval_this_epoch:
#         if val_accs.avg >= best_TA:
#             best_TA = val_accs.avg
#             torch.save(model.state_dict(), os.path.join(save_folder, 'best_SA.pth'))
#         if val_accs_adv.avg >= best_ATA:
#             best_ATA = val_accs_adv.avg
#             torch.save(model.state_dict(), os.path.join(save_folder, 'best_RA.pth'))
#     save_ckpt(epoch, model, optimizer, scheduler, best_TA, best_ATA, training_loss, val_TA, val_ATA, 
#         os.path.join(save_folder, 'latest.pth'))


