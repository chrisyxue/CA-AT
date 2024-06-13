#!/bin/sh

# GRAD_OP = ['avg','thres','orth','thres_layer']
# ADVER_LOSS = ['adver','mart','trades','clp']
# ADVER_CRITERION = ['ce','ce_ls','dlr','dlr_softmax']

#  No WA and Aug on CIFAR10
python train-wa.py --data-dir /data/datasets \
    --log-dir /home/zhiyuxue/results/adver_robustness_notrack \
    --desc test \
    --data cifar10 \
    --batch-size 1024 \
    --model resnet18 \
    --num-adv-epochs 300 \
    --lr 0.4 \
    --unsup-fraction 0.7 \
    --tau 0 \
    --gpu 0 \
    --beta 0.5 \
    --grad_op avg \
    --grad_thres 0.9 \
    --augment False \
    --adver_loss adver \
    --grad_track

python train-wa.py --data-dir /data/datasets \
    --log-dir /home/zhiyuxue/results/adver_robustness_notrack \
    --desc test \
    --data cifar10 \
    --batch-size 1024 \
    --model resnet18 \
    --num-adv-epochs 300 \
    --lr 0.4 \
    --unsup-fraction 0.7 \
    --tau 0 \
    --gpu 0 \
    --beta 1 \
    --grad_op avg \
    --grad_thres 0.9 \
    --augment False \
    --adver_loss adver \
    --grad_track

python train-wa.py --data-dir /data/datasets \
    --log-dir /home/zhiyuxue/results/adver_robustness_notrack \
    --desc test \
    --data cifar10 \
    --batch-size 1024 \
    --model resnet18 \
    --num-adv-epochs 300 \
    --lr 0.4 \
    --unsup-fraction 0.7 \
    --tau 0 \
    --gpu 0 \
    --beta 0 \
    --grad_op thres \
    --grad_thres 0.8 \
    --augment False \
    --adver_loss adver \
    --grad_track

python train-wa.py --data-dir /data/datasets \
    --log-dir /home/zhiyuxue/results/adver_robustness_notrack \
    --desc test \
    --data cifar10 \
    --batch-size 1024 \
    --model resnet18 \
    --num-adv-epochs 300 \
    --lr 0.4 \
    --unsup-fraction 0.7 \
    --tau 0 \
    --gpu 0 \
    --beta 0 \
    --grad_op thres \
    --grad_thres 0.9 \
    --augment False \
    --adver_loss adver \
    --grad_track