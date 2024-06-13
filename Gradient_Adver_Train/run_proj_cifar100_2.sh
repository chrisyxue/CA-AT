#!/bin/sh

# CUDA_VISIBLE_DEVICES=1 python PGDAT_Proj.py --Lambda 0.1 --gpu 1 --grad_proj thres --model WRN-28-10 --grad_thres 0.9 --dataset cifar100
# CUDA_VISIBLE_DEVICES=1 python PGDAT_Proj.py --Lambda 0.1 --gpu 1 --grad_proj thres --model WRN-28-10 --grad_thres 0.8 --dataset cifar100
# CUDA_VISIBLE_DEVICES=1 python PGDAT_Proj.py --Lambda 0.1 --gpu 1 --grad_proj thres --model WRN-28-10 --grad_thres 0.7 --dataset cifar100

# CUDA_VISIBLE_DEVICES=1 python PGDAT_Proj.py --Lambda 0.1 --gpu 1 --grad_proj thres --model WRN-28-10 --grad_thres 0.9 --dataset cifar100 --grad_norm
# CUDA_VISIBLE_DEVICES=1 python PGDAT_Proj.py --Lambda 0.1 --gpu 1 --grad_proj thres --model WRN-28-10 --grad_thres 0.8 --dataset cifar100 --grad_norm
# CUDA_VISIBLE_DEVICES=1 python PGDAT_Proj.py --Lambda 0.1 --gpu 1 --grad_proj thres --model WRN-28-10 --grad_thres 0.7 --dataset cifar100 --grad_norm

CUDA_VISIBLE_DEVICES=1 python PGDAT_Proj.py --Lambda 0.1 --gpu 1 --grad_proj thres --model WRN-28-10 --grad_thres 0.6 --dataset cifar100
CUDA_VISIBLE_DEVICES=1 python PGDAT_Proj.py --Lambda 0.1 --gpu 1 --grad_proj thres --model WRN-28-10 --grad_thres 0.6 --dataset cifar100 --grad_norm
