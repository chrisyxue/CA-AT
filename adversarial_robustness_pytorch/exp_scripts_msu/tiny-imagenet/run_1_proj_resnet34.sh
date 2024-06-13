#!/bin/sh

cd /mnt/home/xuezhiyu/Adver/adversarial_robustness_pytorch

#  No WA and Aug on CIFAR10
# python train-wa.py --data-dir /mnt/home/xuezhiyu/datasets \
#     --log-dir /mnt/home/xuezhiyu/results/adver_robustness_notrack \
#     --desc test \
#     --data tiny-imagenet \
#     --batch-size 1024 \
#     --model preact-resnet34 \
#     --num-adv-epochs 200 \
#     --lr 0.4 \
#     --unsup-fraction 0.7 \
#     --tau 0 \
#     --gpu 0 \
#     --beta 0 \
#     --grad_op thres \
#     --grad_thres 0.9 \
#     --augment False \
#     --grad_track

python train-wa.py --data-dir /mnt/home/xuezhiyu/datasets \
    --log-dir /mnt/home/xuezhiyu/results/adver_robustness_notrack \
    --desc test \
    --data tiny-imagenet \
    --batch-size 1024 \
    --model preact-resnet34 \
    --num-adv-epochs 200 \
    --lr 0.4 \
    --unsup-fraction 0.7 \
    --tau 0 \
    --gpu 0 \
    --beta 0 \
    --grad_op thres \
    --grad_thres 0.8 \
    --augment False \
    --grad_track



