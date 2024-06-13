#!/bin/sh



python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.9 --dataset svhn
python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.95 --dataset svhn
python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.9 --grad_norm --dataset svhn
python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.95 --grad_norm --dataset svhn

python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.9 --use_cat --dataset svhn
python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.95 --use_cat --dataset svhn
python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.9 --grad_norm --use_cat --dataset svhn
python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.95 --grad_norm --use_cat --dataset svhn

python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.9 --use_trades --dataset svhn
python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.95 --use_trades --dataset svhn
python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.9 --grad_norm --use_trades --dataset svhn
python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.95 --grad_norm --use_trades --dataset svhn

python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.9 --use_cat --use_trades --dataset svhn
python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.95 --use_cat --use_trades --dataset svhn
python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.9 --grad_norm --use_cat --use_trades --dataset svhn
python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.95 --grad_norm --use_cat --use_trades --dataset svhn