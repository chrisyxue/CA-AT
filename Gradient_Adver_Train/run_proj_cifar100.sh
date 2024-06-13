#!/bin/sh

python PGDAT_Proj.py --Lambda 0.1 --gpu 1 --grad_proj thres --grad_thres 0.9 --batch_size 128 --dataset cifar100 --grad_norm
python PGDAT_Proj.py --Lambda 0.1 --gpu 1 --grad_proj thres --grad_thres 0.8 --batch_size 128 --dataset cifar100 --grad_norm
python PGDAT_Proj.py --Lambda 0.1 --gpu 1 --grad_proj thres --grad_thres 0.7 --batch_size 128 --dataset cifar100 --grad_norm
python PGDAT_Proj.py --Lambda 0.1 --gpu 1 --grad_proj thres --grad_thres 0.6 --batch_size 128 --dataset cifar100 --grad_norm

python PGDAT_Proj.py --Lambda 0.1 --gpu 1 --grad_proj thres --grad_thres 0.9 --batch_size 128 --dataset cifar100
python PGDAT_Proj.py --Lambda 0.1 --gpu 1 --grad_proj thres --grad_thres 0.8 --batch_size 128 --dataset cifar100
python PGDAT_Proj.py --Lambda 0.1 --gpu 1 --grad_proj thres --grad_thres 0.7 --batch_size 128 --dataset cifar100
python PGDAT_Proj.py --Lambda 0.1 --gpu 1 --grad_proj thres --grad_thres 0.6 --batch_size 128 --dataset cifar100