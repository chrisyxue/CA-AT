#!/bin/sh

cd /scratch/zx1673/codes/Adver/adversarial_robustness_pytorch
#  No WA and No Aug on CIFAR10
# python train-wa.py --data-dir /scratch/zx1673/dataset \
#     --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
#     --desc test \
#     --data cifar10 \
#     --batch-size 1024 \
#     --model resnet34 \
#     --num-adv-epochs 200 \
#     --lr 0.4 \
#     --unsup-fraction 0.7 \
#     --tau 0 \
#     --gpu 0 \
#     --beta 0 \
#     --grad_op thres_layer \
#     --grad_thres 0.9 \
#     --augment False \
#     --grad_track

# python train-wa.py --data-dir /scratch/zx1673/dataset \
#     --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
#     --desc test \
#     --data cifar10 \
#     --batch-size 1024 \
#     --model resnet34 \
#     --num-adv-epochs 200 \
#     --lr 0.4 \
#     --unsup-fraction 0.7 \
#     --tau 0 \
#     --gpu 0 \
#     --beta 0 \
#     --grad_op thres_layer \
#     --grad_thres 0.8 \
#     --augment False \
#     --grad_track

# #  No WA and Aug on CIFAR10
# python train-wa.py --data-dir /scratch/zx1673/dataset \
#     --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
#     --desc test \
#     --data cifar10 \
#     --batch-size 1024 \
#     --model resnet34 \
#     --num-adv-epochs 200 \
#     --lr 0.4 \
#     --unsup-fraction 0.7 \
#     --tau 0 \
#     --gpu 0 \
#     --beta 0 \
#     --grad_op thres_layer \
#     --grad_thres 0.9 \
#     --augment True \
#     --grad_track

# python train-wa.py --data-dir /scratch/zx1673/dataset \
#     --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
#     --desc test \
#     --data cifar10 \
#     --batch-size 1024 \
#     --model resnet34 \
#     --num-adv-epochs 200 \
#     --lr 0.4 \
#     --unsup-fraction 0.7 \
#     --tau 0 \
#     --gpu 0 \
#     --beta 0 \
#     --grad_op thres_layer \
#     --grad_thres 0.8 \
#     --augment True \
#     --grad_track



#  WA and Aug on CIFAR10
python train-wa.py --data-dir /scratch/zx1673/dataset \
    --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
    --desc test \
    --data cifar10 \
    --batch-size 1024 \
    --model resnet34 \
    --num-adv-epochs 200 \
    --lr 0.4 \
    --unsup-fraction 0.7 \
    --tau 0.9 \
    --gpu 0 \
    --beta 0 \
    --grad_op thres_layer \
    --grad_thres 0.9 \
    --augment True \
    --grad_track

python train-wa.py --data-dir /scratch/zx1673/dataset \
    --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
    --desc test \
    --data cifar10 \
    --batch-size 1024 \
    --model resnet34 \
    --num-adv-epochs 200 \
    --lr 0.4 \
    --unsup-fraction 0.7 \
    --tau 0.9 \
    --gpu 0 \
    --beta 0 \
    --grad_op thres_layer \
    --grad_thres 0.8 \
    --augment True \
    --grad_track



python train-wa.py --data-dir /scratch/zx1673/dataset \
    --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
    --desc test \
    --data cifar10 \
    --batch-size 1024 \
    --model resnet34 \
    --num-adv-epochs 200 \
    --lr 0.4 \
    --unsup-fraction 0.7 \
    --tau 0.8 \
    --gpu 0 \
    --beta 0 \
    --grad_op thres_layer \
    --grad_thres 0.9 \
    --augment True \
    --grad_track

python train-wa.py --data-dir /scratch/zx1673/dataset \
    --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
    --desc test \
    --data cifar10 \
    --batch-size 1024 \
    --model resnet34 \
    --num-adv-epochs 200 \
    --lr 0.4 \
    --unsup-fraction 0.7 \
    --tau 0.8 \
    --gpu 0 \
    --beta 0 \
    --grad_op thres_layer \
    --grad_thres 0.8 \
    --augment True \
    --grad_track


python train-wa.py --data-dir /scratch/zx1673/dataset \
    --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
    --desc test \
    --data cifar10 \
    --batch-size 1024 \
    --model resnet34 \
    --num-adv-epochs 200 \
    --lr 0.4 \
    --unsup-fraction 0.7 \
    --tau 0.7 \
    --gpu 0 \
    --beta 0 \
    --grad_op thres_layer \
    --grad_thres 0.9 \
    --augment True \
    --grad_track

python train-wa.py --data-dir /scratch/zx1673/dataset \
    --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
    --desc test \
    --data cifar10 \
    --batch-size 1024 \
    --model resnet34 \
    --num-adv-epochs 200 \
    --lr 0.4 \
    --unsup-fraction 0.7 \
    --tau 0.7 \
    --gpu 0 \
    --beta 0 \
    --grad_op thres_layer \
    --grad_thres 0.8 \
    --augment True \
    --grad_track


