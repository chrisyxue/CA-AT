#!/bin/sh

cd /scratch/zx1673/codes/Adver/adversarial_robustness_pytorch

python train-wa.py --data-dir /scratch/zx1673/dataset \
    --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
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
    --adver_loss clp \
    --grad_thres 0.7 \
    --augment True \
    --attack-eps 8/255