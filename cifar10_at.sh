#!/bin/sh

# <your-data-dir>
# <your-log-dir>

CUDA_VISIBLE_DEVICES=0 python train-wa.py --data-dir <your-data-dir> \
    --log-dir <your-log-dir> \
    --desc test \
    --data cifar10 \
    --batch-size 1024 \
    --model resnet18 \
    --num-adv-epochs 200 \
    --lr 0.4 \
    --tau 0 \
    --gpu 0 \
    --beta 0.5 \
    --grad_op avg \
    --grad_thres 0.7 \
    --augment True \
    --attack-eps 8/255