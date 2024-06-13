#!/bin/sh

cd /mnt/home/xuezhiyu/Adver/adversarial_robustness_pytorch

# No WA and No Aug on CIFAR10


# No WA and Aug

python train-wa.py --data-dir /mnt/home/xuezhiyu/datasets \
    --log-dir /mnt/home/xuezhiyu/results/adver_robustness_notrack \
    --desc test \
    --data cifar10 \
    --batch-size 1024 \
    --model resnet34 \
    --num-adv-epochs 200 \
    --lr 0.4 \
    --unsup-fraction 0.7 \
    --tau 0 \
    --gpu 0 \
    --beta 0.5 \
    --grad_op avg \
    --augment True \
    --grad_track \
    --adver_loss trades \
    --attack-eps 8/255


python train-wa.py --data-dir /mnt/home/xuezhiyu/datasets \
    --log-dir /mnt/home/xuezhiyu/results/adver_robustness_notrack \
    --desc test \
    --data cifar100 \
    --batch-size 1024 \
    --model resnet34 \
    --num-adv-epochs 200 \
    --lr 0.4 \
    --unsup-fraction 0.7 \
    --tau 0 \
    --gpu 0 \
    --beta 0.5 \
    --grad_op avg \
    --augment True \
    --grad_track \
    --adver_loss trades \
    --attack-eps 8/255