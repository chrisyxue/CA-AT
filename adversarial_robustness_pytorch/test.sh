#!/bin/sh

# python train-wa.py --data-dir /scratch/zx1673/dataset \
#     --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
#     --desc test \
#     --data cifar100 \
#     --batch-size 1024 \
#     --model resnet18 \
#     --num-adv-epochs 200 \
#     --lr 0.4 \
#     --unsup-fraction 0.7 \
#     --tau 0 \
#     --gpu 0 \
#     --beta 0 \
#     --grad_op thres \
#     --grad_thres 0.9 \
#     --augment False \
#     --grad_track \
#     --use_ls \
#     --pretrain_type In-Domain
#     # --adaptive_thres StdAccExp \
#     # --use_ls



python train-wa.py --data-dir /scratch/zx1673/dataset \
    --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
    --desc test \
    --data cifar10 \
    --batch-size 1024 \
    --model resnet18 \
    --num-adv-epochs 200 \
    --lr 0.4 \
    --unsup-fraction 0.7 \
    --tau 0 \
    --gpu 0 \
    --beta 0 \
    --grad_op thres \
    --grad_thres 0.9 \
    --augment False \
    --grad_norm \
    --attack-eps 8/255 \
    --grad_track
