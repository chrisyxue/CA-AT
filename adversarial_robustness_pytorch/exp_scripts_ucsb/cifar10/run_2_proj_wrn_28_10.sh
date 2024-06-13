#!/bin/sh

cd /home/zhiyu/codes/Adver/adversarial_robustness_pytorch


# wrn-28-10

CUDA_VISIBLE_DEVICES=5 python train-wa.py --data-dir /data/zhiyu/dataset \
    --log-dir /data/zhiyu/results/xuezhiyu/adver_robustness_notrack \
    --desc test \
    --data cifar10 \
    --batch-size 1024 \
    --model wrn-28-10 \
    --adver_loss mart \
    --num-adv-epochs 200 \
    --lr 0.4 \
    --unsup-fraction 0.7 \
    --tau 0 \
    --gpu 0 \
    --beta 0.25 \
    --grad_op avg \
    --grad_thres 0 \
    --augment True \
    --attack-eps 8/255

CUDA_VISIBLE_DEVICES=5 python train-wa.py --data-dir /data/zhiyu/dataset \
    --log-dir /data/zhiyu/results/xuezhiyu/adver_robustness_notrack \
    --desc test \
    --data cifar10 \
    --batch-size 1024 \
    --model wrn-28-10 \
    --adver_loss mart \
    --num-adv-epochs 200 \
    --lr 0.4 \
    --unsup-fraction 0.7 \
    --tau 0 \
    --gpu 0 \
    --beta 0.5 \
    --grad_op avg \
    --grad_thres 0 \
    --augment True \
    --attack-eps 8/255

CUDA_VISIBLE_DEVICES=5 python train-wa.py --data-dir /data/zhiyu/dataset \
    --log-dir /data/zhiyu/results/xuezhiyu/adver_robustness_notrack \
    --desc test \
    --data cifar10 \
    --batch-size 1024 \
    --model wrn-28-10 \
    --adver_loss mart \
    --num-adv-epochs 200 \
    --lr 0.4 \
    --unsup-fraction 0.7 \
    --tau 0 \
    --gpu 0 \
    --beta 0.75 \
    --grad_op avg \
    --grad_thres 0 \
    --augment True \
    --attack-eps 8/255