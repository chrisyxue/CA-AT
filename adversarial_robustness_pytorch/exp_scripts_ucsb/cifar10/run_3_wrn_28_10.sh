#!/bin/sh

cd /scratch/zx1673/codes/Adver/adversarial_robustness_pytorch


# wrn-28-10
python train-wa.py --data-dir /scratch/zx1673/dataset \
    --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
    --desc test \
    --data cifar10 \
    --batch-size 512 \
    --model wrn-28-10 \
    --num-adv-epochs 200 \
    --lr 0.4 \
    --unsup-fraction 0.7 \
    --tau 0 \
    --gpu 0 \
    --beta 1 \
    --grad_op avg \
    --grad_thres 0.7 \
    --augment True \
    --attack-eps 8/255