"""
Evaluation with AutoAttack.
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
import pdb


from attack_test import eval_untarget_pgd

# Setup

parse = parser_eval()
args = parse.parse_args()
DATA_DIR = args.data_dir

# LOG_DIR = args.log_dir + args.desc
LOG_DIR = args.desc
with open(LOG_DIR+'/args.txt', 'r') as f:
    old = json.load(f)
    args.__dict__ = dict(vars(args), **old)
DATA_DIR = DATA_DIR + '/' + args.data + '/'
WEIGHTS = LOG_DIR + '/weights-best.pt'
log_path = LOG_DIR + '/log-untarget.log'
logger = Logger(log_path)

info = get_data_info(DATA_DIR)
BATCH_SIZE = args.batch_size
BATCH_SIZE_VALIDATION = args.batch_size_validation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.log('Using device: {}'.format(device))



# Load data

seed(args.seed)
# pdb.set_trace()
_, _, train_dataloader, test_dataloader = load_data(args, DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=False, 
                                                    shuffle_train=False,num_workers=3)

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

if 'cosine' in args.log_dir:
    args.classifier = 'cosine'
else:
    args.classifier = 'linear'

# Model
model = create_model(args.model, args.normalize, info, device, args)
checkpoint = torch.load(WEIGHTS)
if 'tau' in args and args.tau:
    print ('Using WA model.')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
del checkpoint



# AA Evaluation

seed(args.seed)
norm = 'Linf' if args.attack in ['fgsm', 'linf-pgd', 'linf-df'] else 'L2'

# eps_inf_lst = [8/255,16/255,24/255,32/255]
# eps_l2_lst = [0.5,1,1.5,2]
eps_inf_lst = [8/255]
eps_l2_lst = [0.5]
res_lst = []
for i in range(len(eps_inf_lst)):
    eps_inf = eps_inf_lst[i]
    eps_l2 = eps_l2_lst[i]
    log_path = LOG_DIR + '/log-pgd-'+str(int(eps_inf*255))+'.log'
    res = eval_untarget_pgd(model, test_dataloader, eps_inf, eps_l2)
    res_lst.append(res)
    res_df = pd.DataFrame(res_lst)
    res_df.to_csv(os.path.join(LOG_DIR,'pgd-attacks-eval2.csv'))


print('Script Completed.')