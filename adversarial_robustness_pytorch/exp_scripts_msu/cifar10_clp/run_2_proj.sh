#!/bin/sh

cd /mnt/home/xuezhiyu/Adver/adversarial_robustness_pytorch


#  No WA and Aug on CIFAR10
python train-wa.py --data-dir /mnt/home/xuezhiyu/datasets \
    --log-dir /mnt/home/xuezhiyu/results/adver_robustness_notrack \
    --desc test \
    --data cifar100 \
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
    --augment True \
    --grad_track \
    --adver_loss clp \
    --attack-eps 8/255

python train-wa.py --data-dir /mnt/home/xuezhiyu/datasets \
    --log-dir /mnt/home/xuezhiyu/results/adver_robustness_notrack \
    --desc test \
    --data cifar100 \
    --batch-size 1024 \
    --model resnet18 \
    --num-adv-epochs 200 \
    --lr 0.4 \
    --unsup-fraction 0.7 \
    --tau 0 \
    --gpu 0 \
    --beta 0 \
    --grad_op thres \
    --grad_thres 0.8 \
    --augment True \
    --grad_track \
    --adver_loss clp \
    --attack-eps 8/255
