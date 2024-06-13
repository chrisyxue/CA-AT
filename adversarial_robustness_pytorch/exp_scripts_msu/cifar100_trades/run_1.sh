#!/bin/sh

cd /scratch/zx1673/codes/Adver/adversarial_robustness_pytorch

# No WA and No Aug on CIFAR10
python train-wa.py --data-dir /scratch/zx1673/dataset \
    --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
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
    --grad_op avg \
    --augment False \
    --grad_track \
    --adver_loss trades \
    --attack-eps 8/255 

python train-wa.py --data-dir /scratch/zx1673/dataset \
    --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
    --desc test \
    --data cifar100 \
    --batch-size 1024 \
    --model resnet18 \
    --num-adv-epochs 200 \
    --lr 0.4 \
    --unsup-fraction 0.7 \
    --tau 0 \
    --gpu 0 \
    --beta 0.5 \
    --grad_op avg \
    --augment False \
    --grad_track \
    --adver_loss trades \
    --attack-eps 8/255

python train-wa.py --data-dir /scratch/zx1673/dataset \
    --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
    --desc test \
    --data cifar100 \
    --batch-size 1024 \
    --model resnet18 \
    --num-adv-epochs 200 \
    --lr 0.4 \
    --unsup-fraction 0.7 \
    --tau 0 \
    --gpu 0 \
    --beta 1 \
    --grad_op avg \
    --augment False \
    --grad_track \
    --adver_loss trades \
    --attack-eps 8/255


# No WA and Aug

python train-wa.py --data-dir /scratch/zx1673/dataset \
    --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
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
    --grad_op avg \
    --augment True \
    --grad_track \
    --adver_loss trades \
    --attack-eps 8/255 

python train-wa.py --data-dir /scratch/zx1673/dataset \
    --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
    --desc test \
    --data cifar100 \
    --batch-size 1024 \
    --model resnet18 \
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

python train-wa.py --data-dir /scratch/zx1673/dataset \
    --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
    --desc test \
    --data cifar100 \
    --batch-size 1024 \
    --model resnet18 \
    --num-adv-epochs 200 \
    --lr 0.4 \
    --unsup-fraction 0.7 \
    --tau 0 \
    --gpu 0 \
    --beta 1 \
    --grad_op avg \
    --augment True \
    --grad_track \
    --adver_loss trades \
    --attack-eps 8/255

# WA and Aug
python train-wa.py --data-dir /scratch/zx1673/dataset \
    --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
    --desc test \
    --data cifar100 \
    --batch-size 1024 \
    --model resnet18 \
    --num-adv-epochs 200 \
    --lr 0.4 \
    --unsup-fraction 0.7 \
    --tau 0.995 \
    --gpu 0 \
    --beta 0 \
    --grad_op avg \
    --augment True \
    --grad_track \
    --adver_loss trades \
    --attack-eps 8/255 

python train-wa.py --data-dir /scratch/zx1673/dataset \
    --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
    --desc test \
    --data cifar100 \
    --batch-size 1024 \
    --model resnet18 \
    --num-adv-epochs 200 \
    --lr 0.4 \
    --unsup-fraction 0.7 \
    --tau 0.995 \
    --gpu 0 \
    --beta 0.5 \
    --grad_op avg \
    --augment True \
    --grad_track \
    --adver_loss trades \
    --attack-eps 8/255

python train-wa.py --data-dir /scratch/zx1673/dataset \
    --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
    --desc test \
    --data cifar100 \
    --batch-size 1024 \
    --model resnet18 \
    --num-adv-epochs 200 \
    --lr 0.4 \
    --unsup-fraction 0.7 \
    --tau 0.995 \
    --gpu 0 \
    --beta 1 \
    --grad_op avg \
    --augment Tru--adver_loss trades \
    --attack-eps 8/255