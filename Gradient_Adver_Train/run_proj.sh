#!/bin/sh

# python PGDAT_Proj.py --Lambda 0.1 --gpu 3 --grad_norm
# python PGDAT_Proj.py --Lambda 0.3 --gpu 3 --grad_norm
# python PGDAT_Proj.py --Lambda 0.5 --gpu 3 --grad_norm
# python PGDAT_Proj.py --Lambda 0.7 --gpu 3 --grad_norm
# python PGDAT.py --Lambda 0.5 --gpu 0 --

# python PGDAT_Proj.py --Lambda 0.1 --gpu 0 --grad_proj thres --grad_thres 0.8 --use_pretrain
# python PGDAT_Proj.py --Lambda 0.1 --gpu 0 --grad_proj thres --grad_thres 0.9 --use_pretrain
# python PGDAT_Proj.py --Lambda 0.1 --gpu 0 --grad_proj thres --grad_thres 0.95 --use_pretrain
# python PGDAT_Proj.py --Lambda 0.1 --gpu 0 --grad_proj thres --grad_thres 0.95

# python PGDAT_Proj.py --Lambda 0.1 --gpu 0 --grad_proj thres --grad_thres 0.9 --use_pretrain --steps 10
# python PGDAT_Proj.py --Lambda 0.1 --gpu 0 --grad_proj thres --grad_thres 0.9 --use_pretrain --steps 15
# python PGDAT_Proj.py --Lambda 0.1 --gpu 0 --grad_proj thres --grad_thres 0.9 --use_pretrain --steps 20


# python PGDAT.py --Lambda 0 --gpu 3
# python PGDAT_Proj.py --Lambda 0 --gpu 3 --grad_norm

python PGDAT.py --Lambda 0 --gpu 3 --dataset svhn
python PGDAT.py --Lambda 0.5 --gpu 3 --dataset svhn

python PGDAT_Proj.py --Lambda 0.1 --gpu 3 --grad_proj thres --grad_thres 0.9 --dataset svhn
python PGDAT_Proj.py --Lambda 0.1 --gpu 3 --grad_proj thres --grad_thres 0.95 --dataset svhn
python PGDAT_Proj.py --Lambda 0.1 --gpu 3 --grad_proj thres --grad_thres 0.9 --grad_norm --dataset svhn
python PGDAT_Proj.py --Lambda 0.1 --gpu 3 --grad_proj thres --grad_thres 0.95 --grad_norm --dataset svhn

python PGDAT_Proj.py --Lambda 0.1 --gpu 3 --grad_proj thres --grad_thres 0.9 --use_cat --dataset svhn
python PGDAT_Proj.py --Lambda 0.1 --gpu 3 --grad_proj thres --grad_thres 0.95 --use_cat --dataset svhn
python PGDAT_Proj.py --Lambda 0.1 --gpu 3 --grad_proj thres --grad_thres 0.9 --grad_norm --use_cat --dataset svhn
python PGDAT_Proj.py --Lambda 0.1 --gpu 3 --grad_proj thres --grad_thres 0.95 --grad_norm --use_cat --dataset svhn

python PGDAT_Proj.py --Lambda 0.1 --gpu 3 --grad_proj thres --grad_thres 0.9 --use_trades --dataset svhn
python PGDAT_Proj.py --Lambda 0.1 --gpu 3 --grad_proj thres --grad_thres 0.95 --use_trades --dataset svhn
python PGDAT_Proj.py --Lambda 0.1 --gpu 3 --grad_proj thres --grad_thres 0.9 --grad_norm --use_trades --dataset svhn
python PGDAT_Proj.py --Lambda 0.1 --gpu 3 --grad_proj thres --grad_thres 0.95 --grad_norm --use_trades --dataset svhn

python PGDAT_Proj.py --Lambda 0.1 --gpu 3 --grad_proj thres --grad_thres 0.9 --use_cat --use_trades --dataset svhn
python PGDAT_Proj.py --Lambda 0.1 --gpu 3 --grad_proj thres --grad_thres 0.95 --use_cat --use_trades --dataset svhn
python PGDAT_Proj.py --Lambda 0.1 --gpu 3 --grad_proj thres --grad_thres 0.9 --grad_norm --use_cat --use_trades --dataset svhn
python PGDAT_Proj.py --Lambda 0.1 --gpu 3 --grad_proj thres --grad_thres 0.95 --grad_norm --use_cat --use_trades --dataset svhn
