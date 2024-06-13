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
log_path = LOG_DIR + '/log-aa.log'
logger = Logger(log_path)


info = get_data_info(DATA_DIR)
BATCH_SIZE = args.batch_size
BATCH_SIZE_VALIDATION = args.batch_size_validation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.log('Using device: {}'.format(device))



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

# eps_lst = [8/255,16/255,24/255,32/255]
eps_lst = [8/255]

for eps in eps_lst:
    log_path = LOG_DIR + '/log-aa-'+str(int(eps*255))+'.log'
    adversary = AutoAttack(model, norm=norm, eps=eps, log_path=log_path, version=args.version, seed=args.seed)
    if args.version == 'custom':
        adversary.attacks_to_run = ['apgd-ce', 'apgd-t']
        adversary.apgd.n_restarts = 1
        adversary.apgd_targeted.n_restarts = 1
    
    with torch.no_grad():
        x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=BATCH_SIZE_VALIDATION)



print ('Script Completed.')