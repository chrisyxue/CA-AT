'''
PGDAT

Generate Adver samples in different epoches in adversarial training

(Also store models for different epoches)

'''

import os, argparse, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam, lr_scheduler

from models.cifar10.resnet import ResNet34, ResNet34_ENC, ResNet_CLS
from models.svhn.wide_resnet import WRN16_8
from models.stl10.wide_resnet import WRN40_2

from dataloaders.cifar10 import cifar10_dataloaders
from dataloaders.svhn import svhn_dataloaders
from dataloaders.stl10 import stl10_dataloaders

from utils.utils import *
from utils.context import ctx_noparamgrad_and_eval
from attacks.pgd import PGD

from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision import transforms


parser = argparse.ArgumentParser(description='CIFAR10 Training against DDN Attack')
parser.add_argument('--gpu', default='0')
parser.add_argument('--cpus', default=4)
# dataset:
parser.add_argument('--dataset', '--ds', default='cifar10', choices=['cifar10', 'svhn', 'stl10'], help='which dataset to use')
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

parser.add_argument('--val_type', default='val')
args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.backends.cudnn.benchmark = True

# data loader:
if args.dataset == 'cifar10':
    train_loader, val_loader, _ = cifar10_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus)
elif args.dataset == 'svhn':
    train_loader, val_loader, _ = svhn_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus)
elif args.dataset == 'stl10':
    train_loader, val_loader = stl10_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus)

# model:
if args.dataset == 'cifar10':
    model_fn = ResNet34
    # model_enc = ResNet34_ENC()
    # model_cls = ResNet_CLS()
elif args.dataset == 'svhn':
    model_fn = WRN16_8
elif args.dataset == 'stl10':
    model_fn = WRN40_2
model = model_fn().cuda()
model = torch.nn.DataParallel(model)

def store_data(imgs_adv,labels, data_path, count, data_type='train'):
    label_file = open(os.path.join(data_path, 'labels.txt'), 'a+')

    # train or val
    data_path = os.path.join(data_path,data_type)
    for idx in range(imgs.shape[0]):
    
        img_adv = imgs_adv[idx]
        img_adv = img_adv.cpu().detach()
        label = labels.cpu().numpy()[idx]
        
        data_path_class = os.path.join(data_path,str(label))
        create_dir(data_path_class)

        img_adv_name = os.path.join(data_path_class,'img_adv' + '_' + str(label) + '_' + str(count)+ '.png')
        save_image(img_adv, img_adv_name)

        label_file.write(img_adv_name + ' ' + str(label) + '\n')
        count = count + 1
# mkdirs:
model_str = model_fn.__name__
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

save_folder = os.path.join('/localscratch2/xuezhiyu/Adver_Forget', args.dataset, model_str, '%s_%s' % (attack_str, loss_str))
save_folder_models = os.path.join(save_folder,'models')
save_folder_data = os.path.join(save_folder,'adver_data')

res_path = os.path.join('/localscratch2/xuezhiyu/Adver_Forget_Res', args.dataset, model_str, '%s_%s' % (attack_str, loss_str))
create_dir(res_path)
res_path = os.path.join(res_path,'res.csv')

model_data_dic = {}

for i in os.listdir(save_folder_models):
    epoch_num = i.split('_')[-1]
    model_data_dic[epoch_num] = ['','']
    model_data_dic[epoch_num][0] = os.path.join(save_folder_models,i,epoch_num+'.pth')

for i in os.listdir(save_folder_data):
    epoch_num = i.split('_')[-1]
    model_data_dic[epoch_num][1] = os.path.join(save_folder_data,i,args.val_type)

# save_folder_models_list = [os.path.join(save_folder_models,i) for i in os.listdir(save_folder_models)]
# save_folder_data_list = [os.path.join(save_folder_data,i,args.val_type) for i in os.listdir(save_folder_data)]
# print(save_folder_models_list)


# dataset and loader
transform = transforms.Compose([
        transforms.ToTensor(),
    ])
cols = [int(i) for i in list(model_data_dic.keys())]
cols.sort()
res_data = pd.DataFrame(columns = cols,index = cols)


print(res_data)

for model_idx in model_data_dic.keys():
    model_path = model_data_dic[model_idx][0]
    model.load_state_dict(torch.load(model_path))
    
    for data_idx in model_data_dic.keys():
        data_path = model_data_dic[data_idx][1]
        dataset = ImageFolder(data_path,transform=transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True, drop_last=False, num_workers=4)
        accs = AverageMeter()

        for i, (imgs, labels) in enumerate(data_loader):
            imgs, labels = imgs.cuda(), labels.cuda()
            logits = model(imgs)
            accs.append((logits.argmax(1) == labels).float().mean().item())
        
        print(model_idx,data_idx,accs.avg)
        res_data.loc[int(model_idx),int(data_idx)] = accs.avg
    
    print(res_data)

res_data.to_csv(res_path)


