"""
Evaluation For the representation space
"""

import json
import time
import argparse
import shutil

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from autoattack import AutoAttack
    
from core.data import get_data_info
from core.data import load_data
from core.models import create_model

from core.utils import Logger
from core.utils import parser_eval
from core.utils import seed
from core.utils import rep_analysis
import pdb

from attack_test import eval_target_adv_test
# from attack_test import autopgd_whitebox, tapgd_whitebox
import core.attacks.torchattackslib.torchattacks as torchattacks
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Setup
parse = parser_eval()
args = parse.parse_args()
DATA_DIR = args.data_dir

# LOG_DIR = args.log_dir + args.desc
LOG_DIR = args.desc
with open(LOG_DIR+'/args.txt', 'r') as f:
    old = json.load(f)
    args.__dict__ = dict(vars(args), **old)

if args.data == 'cifar10':
    n_classes = 10
elif args.data == 'cifar100':
    n_classes = 100
else:
    raise ValueError('No Data named '+str(args.data))

DATA_DIR = DATA_DIR + '/' + args.data + '/'
WEIGHTS = LOG_DIR + '/weights-best.pt'
log_path = LOG_DIR + '/log-rep.log'
logger = Logger(log_path)

info = get_data_info(DATA_DIR)
BATCH_SIZE = args.batch_size
BATCH_SIZE_VALIDATION = args.batch_size_validation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.log('Using device: {}'.format(device))



def adver_rep_analysis(model,loader,epsilon=8/255,target_attack=True):
    from torch.autograd import Variable
    model.eval()

    clean_rep_lst = []
    adv_rep_lst = []
    clean_logit_lst = []
    adv_logit_lst = []
    y_lst = []
    for data, target in loader:
        data, target = data.cuda(), target.cuda()
        X, y = Variable(data, requires_grad=True), Variable(target)
        # rep = model.encode(X,normalize=False)
        rep_clean = model.module[0].encode(X,normalize=False)
        logit_clean = model.module[0](X)

        if target_attack:
            atk = torchattacks.APGD(model, norm='Linf', eps=epsilon)
            X_adv = atk(X, y)
        else:
            atk =  torchattacks.APGDT(model, norm='Linf', eps=epsilon, n_classes=n_classes)
            X_adv = atk(X, y)
        
        rep_adv = model.module[0].encode(X_adv,normalize=False)
        logit_adv = model.module[0](X_adv)

        clean_rep_lst.append(rep_clean.detach().cpu())
        adv_rep_lst.append(rep_adv.detach().cpu())
        clean_logit_lst.append(logit_clean.detach().cpu())
        adv_logit_lst.append(logit_adv.detach().cpu())
        y_lst.append(y.detach().cpu())

    clean_rep_all = torch.concat(clean_rep_lst)
    adv_rep_all = torch.concat(adv_rep_lst)
    clean_logit_all = torch.concat(clean_logit_lst)
    adv_logit_all = torch.concat(adv_logit_lst)
    y_all = torch.concat(y_lst)
    
    # sample attack successfully
    adv_rep_all_suc = adv_rep_all[adv_logit_all.data.max(1)[1]!=y_all]
    clean_rep_all_suc = clean_rep_all[adv_logit_all.data.max(1)[1]!=y_all]

    y_all_suc = y_all[adv_logit_all.data.max(1)[1]!=y_all]
    ins_class_metrics_clean_adv, inter_class_metrics_clean_adv, ins_class_metrics_adv_clean, inter_class_metrics_adv_clean, class_var_clean, class_var_adv = rep_analysis.two_group_rep_class_wise_analysis(rep_all_a=clean_rep_all_suc,rep_all_b=adv_rep_all_suc,y_all=y_all_suc)
    ins_class_metrics_clean, inter_class_metrics_clean, _, class_centers_clean = rep_analysis.rep_class_wise_analysis(clean_rep_all_suc,y_all_suc)
    ins_class_metrics_adv, inter_class_metrics_adv, _, class_centers_adv = rep_analysis.rep_class_wise_analysis(adv_rep_all_suc,y_all_suc)
    suc_metric = [ins_class_metrics_adv_clean,inter_class_metrics_adv_clean,ins_class_metrics_clean,inter_class_metrics_clean,ins_class_metrics_adv, inter_class_metrics_adv]

    # sample attack unsuccessfully
    adv_rep_all_suc = adv_rep_all[adv_logit_all.data.max(1)[1]==y_all]
    clean_rep_all_suc = clean_rep_all[adv_logit_all.data.max(1)[1]==y_all]
    
    y_all_unsuc = y_all[adv_logit_all.data.max(1)[1]==y_all]
    ins_class_metrics_clean_adv, inter_class_metrics_clean_adv, ins_class_metrics_adv_clean, inter_class_metrics_adv_clean, class_var_clean, class_var_adv = rep_analysis.two_group_rep_class_wise_analysis(rep_all_a=clean_rep_all_suc,rep_all_b=adv_rep_all_suc,y_all=y_all_unsuc)
    ins_class_metrics_clean, inter_class_metrics_clean, _, class_centers_clean = rep_analysis.rep_class_wise_analysis(clean_rep_all_suc,y_all_unsuc)
    ins_class_metrics_adv, inter_class_metrics_adv, _, class_centers_adv = rep_analysis.rep_class_wise_analysis(adv_rep_all_suc,y_all_unsuc)
    unsuc_metric = [ins_class_metrics_adv_clean,inter_class_metrics_adv_clean,ins_class_metrics_clean,inter_class_metrics_clean,ins_class_metrics_adv, inter_class_metrics_adv]
    
    name = ['ins_class_metrics_adv_clean','inter_class_metrics_adv_clean','ins_class_metrics_clean','inter_class_metrics_clean','ins_class_metrics_adv', 'inter_class_metrics_adv']
    return name, suc_metric, unsuc_metric

# Load data

seed(args.seed)
_, _, train_dataloader, test_dataloader = load_data(DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=False, 
                                                    shuffle_train=False)





if args.train:
    logger.log('Evaluating on training set.')
    l = [x for (x, y) in train_dataloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in train_dataloader]
    y_test = torch.cat(l, 0)
else:
    l = [x for (x, y) in test_dataloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_dataloader]
    y_test = torch.cat(l, 0)



# Model
model = create_model(args.model, args.normalize, info, device)
checkpoint = torch.load(WEIGHTS)
if 'tau' in args and args.tau:
    print ('Using WA model.')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
del checkpoint



# AA Evaluation

seed(args.seed)
norm = 'Linf' if args.attack in ['fgsm', 'linf-pgd', 'linf-df'] else 'L2'

eps_inf_lst = [8/255,16/255,24/255,32/255]
eps_l2_lst = [0.5,1,1.5,2]
res_lst = []
# for i in range(len(eps_inf_lst)):
#     eps_inf = eps_inf_lst[i]
#     eps_l2 = eps_l2_lst[i]
#     log_path = LOG_DIR + '/log-untarget-'+str(int(eps_inf*255))+'.log'

epsilon = 8/255
name, suc_metric, unsuc_metric = adver_rep_analysis(model=model,loader=test_dataloader,epsilon=epsilon,target_attack=True)
table_dir = os.path.join(LOG_DIR,'tables','rep_analysis')
if os.path.exists(table_dir) is False:
    os.makedirs(table_dir)

for j in range(len(name)):
    # title  = name[j] + '_' + str(epsilon)
    title  = name[j]
    data = pd.concat([pd.DataFrame(suc_metric[j]),pd.DataFrame(unsuc_metric[j])],axis=1)
    data.columns = ['L2_Distance_Suc','Linf_Distance_Suc','Cos_Suc','L2_Distance_Unsuc','Linf_Distance_Unsuc','Cos_Unsuc']
    
    data.to_csv(os.path.join(table_dir,title+'.csv'))
    # res = eval_target_adv_test(model, test_dataloader, eps_inf, eps_l2, n_classes)
    # res_lst.append(res)
    # res_df = pd.DataFrame(res_lst)
    # res_df.to_csv(os.path.join(LOG_DIR,'target-attacks-eval.csv'))

print ('Script Completed.')