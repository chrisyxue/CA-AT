#!/bin/sh

cd /home/zhiyu/codes/Adver/adversarial_robustness_pytorch


CUDA_VISIBLE_DEVICES=3 python train-wa.py --data-dir /data/zhiyu/dataset \
    --log-dir /data/zhiyu/results/xuezhiyu/adver_robustness_notrack \
    --desc test \
    --data cifar10 \
    --batch-size 1024 \
    --model wrn-28-10 \
    --num-adv-epochs 200 \
    --lr 0.4 \
    --unsup-fraction 0.7 \
    --tau 0 \
    --gpu 0 \
    --beta 0 \
    --grad_op thres \
    --grad_thres 0.65 \
    --augment True \
    --attack-eps 8/255

CUDA_VISIBLE_DEVICES=3 python train-wa.py --data-dir /data/zhiyu/dataset \
    --log-dir /data/zhiyu/results/xuezhiyu/adver_robustness_notrack \
    --desc test \
    --data cifar10 \
    --batch-size 1024 \
    --model wrn-28-10 \
    --num-adv-epochs 200 \
    --lr 0.4 \
    --unsup-fraction 0.7 \
    --tau 0 \
    --gpu 0 \
    --beta 0 \
    --grad_op thres \
    --grad_thres 0.6 \
    --augment True \
    --attack-eps 8/255