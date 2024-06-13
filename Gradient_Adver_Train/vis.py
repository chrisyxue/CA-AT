'''
PGDAT
'''

import os, argparse, time
import numpy as np
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
args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.backends.cudnn.benchmark = True

def store_imgs(imgs,imgs_adv,labels, vis_save_folder, count):
    # count = count + imgs.shape[0]
    for idx in range(imgs.shape[0]):
        img,img_adv = imgs[idx],imgs_adv[idx]
        dif = (img.cpu().detach().numpy() - img_adv.cpu().detach().numpy()).mean(0)
        dif_origin = (img.cpu().detach().numpy() - img_adv.cpu().detach().numpy())
        img = img.mean(0).cpu().detach().numpy()
        img_adv = img_adv.mean(0).cpu().detach().numpy()
        label = labels.cpu().numpy()[idx]
        img_name = 'img'  + '_' + str(label) + '_' + str(count)+ '.png'
        img_adv_name = 'img'  + '_' + str(label) + '_' + str(count)+ '.png'
        dif_name = 'dif' + '_' + str(label) + '_' + str(count) + '.npy'
        fig = plt.figure()
        
        a = fig.add_subplot(1, 3, 1)
        plt.imshow(img)
        a.axis('off')
        a.set_title('origin')
        # plt.savefig(os.path.join(vis_save_folder,img_name))
        
        a = fig.add_subplot(1, 3, 2)
        plt.imshow(img_adv)
        a.axis('off')
        a.set_title('adv')

        a = fig.add_subplot(1, 3, 3)
        plt.imshow(dif)
        a.axis('off')
        a.set_title('dif')

        plt.savefig(os.path.join(vis_save_folder,img_name))
        np.save(os.path.join(vis_save_folder,dif_name),dif)

        # print(img.shape)
        count = count + 1

def store_origin_data(imgs,imgs_adv,labels, origin_data_folder, count):
    for idx in range(imgs.shape[0]):
        img,img_adv = imgs[idx],imgs_adv[idx]
        dif = (img.cpu().detach().numpy() - img_adv.cpu().detach().numpy())
        img = img.cpu().detach().numpy()
        img_adv = img_adv.cpu().detach().numpy()
        label = labels.cpu().numpy()[idx]
        img_name = 'img'  + '_' + str(label) + '_' + str(count)+ '.png'
        img_adv_name = 'img_adv'  + '_' + str(label) + '_' + str(count)+ '.png'
        dif_name = 'dif' + '_' + str(label) + '_' + str(count) + '.npy'

        np.save(os.path.join(origin_data_folder,dif_name),img)
        np.save(os.path.join(origin_data_folder,img_adv_name),img_adv)
        np.save(os.path.join(origin_data_folder,dif_name),dif)

        count = count + 1

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
save_folder = os.path.join('/localscratch/xuezhiyu/adver-robust/Once-for-All-Adversarial-Training/Results/Normal', args.dataset, model_str, '%s_%s' % (attack_str, loss_str))

vis_save_folder = os.path.join('/localscratch2/xuezhiyu/Adver_Data_Vis', args.dataset, model_str, '%s_%s' % (attack_str, loss_str))
origin_data_folder = os.path.join('/localscratch2/xuezhiyu/Adver_Data', args.dataset, model_str, '%s_%s' % (attack_str, loss_str))
create_dir(vis_save_folder)
create_dir(origin_data_folder)

if os.path.exists(save_folder):
    model_path = os.path.join(save_folder,'best_SA.pth')
    model.load_state_dict(torch.load(model_path))
else:
    raise ValueError("No such folder %s" %(save_folder))

# attacker:
attacker = PGD(eps=args.eps/255, steps=args.steps)

## training:
for epoch in range(1):
    val_accs, val_accs_adv = AverageMeter(), AverageMeter()
    count = 0
    for i, (imgs, labels) in enumerate(val_loader):
        imgs, labels = imgs.cuda(), labels.cuda()

        # generate adversarial images:
        with ctx_noparamgrad_and_eval(model):
            imgs_adv = attacker.attack(model, imgs, labels)
        linf_norms = (imgs_adv - imgs).view(imgs.size()[0], -1).norm(p=np.Inf, dim=1)
        logits_adv = model(imgs_adv.detach())
        # logits for clean imgs:
        logits = model(imgs)
        # print(imgs_adv.shape)
        # store_imgs(imgs, imgs_adv, labels)
        # print(model)
        vis_save_folder_origin = os.path.join(vis_save_folder,'origin')
        create_dir(vis_save_folder_origin)
        origin_data_folder_origin = os.path.join(origin_data_folder,'origin')
        create_dir(origin_data_folder_origin)

        store_imgs(imgs, imgs_adv, labels, vis_save_folder_origin, count)
        store_origin_data(imgs, imgs_adv, labels, origin_data_folder_origin, count)
        if args.dataset == 'cifar10':
            vis_save_folder_1 = os.path.join(vis_save_folder,'fm1')
            create_dir(vis_save_folder_1)
            origin_data_folder_1 = os.path.join(origin_data_folder,'fm1')
            create_dir(origin_data_folder_1)
            fm1 = F.relu(model.module.bn1(model.module.conv1(imgs)))
            fm1_adv = F.relu(model.module.bn1(model.module.conv1(imgs_adv)))
            store_imgs(fm1, fm1_adv, labels, vis_save_folder_1, count)
            store_origin_data(fm1, fm1_adv, labels, origin_data_folder_1, count)
            
            vis_save_folder_2 = os.path.join(vis_save_folder,'fm2')
            create_dir(vis_save_folder_2)
            origin_data_folder_2 = os.path.join(origin_data_folder,'fm2')
            create_dir(origin_data_folder_2)
            fm2 = model.module.layer1(fm1)
            fm2_adv = model.module.layer1(fm1_adv)
            store_imgs(fm2, fm2_adv, labels, vis_save_folder_2, count)
            store_origin_data(fm2, fm2_adv, labels, origin_data_folder_2, count)

            vis_save_folder_3 = os.path.join(vis_save_folder,'fm3')
            create_dir(vis_save_folder_3)
            origin_data_folder_3 = os.path.join(origin_data_folder,'fm3')
            create_dir(origin_data_folder_3)
            fm3 = model.module.layer2(fm2)
            fm3_adv = model.module.layer2(fm2_adv)
            store_imgs(fm3, fm3_adv, labels, vis_save_folder_3, count)
            store_origin_data(fm3, fm3_adv, labels, origin_data_folder_3, count)

            vis_save_folder_4 = os.path.join(vis_save_folder,'fm4')
            create_dir(vis_save_folder_4)
            origin_data_folder_4 = os.path.join(origin_data_folder,'fm4')
            create_dir(origin_data_folder_4)
            fm4 = model.module.layer3(fm3)
            fm4_adv = model.module.layer3(fm3_adv)
            store_imgs(fm4, fm4_adv, labels, vis_save_folder_4, count)
            store_origin_data(fm4, fm4_adv, labels, origin_data_folder_4, count)


            vis_save_folder_5 = os.path.join(vis_save_folder,'fm5')
            create_dir(vis_save_folder_5)
            origin_data_folder_5 = os.path.join(origin_data_folder,'fm5')
            create_dir(origin_data_folder_5)
            fm5 = model.module.layer4(fm4)
            fm5_adv = model.module.layer4(fm4_adv)
            store_imgs(fm5, fm5_adv, labels, vis_save_folder_5, count)
            store_origin_data(fm5, fm5_adv, labels, origin_data_folder_5, count)
            
            # store_imgs(imgs,imgs_adv,labels, vis_save_folder)
            print(fm1.shape)

        count = count + imgs.shape[0]

        if count>600:
            break
        

