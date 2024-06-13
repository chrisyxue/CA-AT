#!/bin/sh


# # Baseline
python eval-untarget.py --wb --data-dir /scratch/zx1673/dataset \
    --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
    --desc /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack/cifar100_aug/1_resnet18_clp_thres_thres_0.9/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4/

python eval-untarget.py --wb --data-dir /scratch/zx1673/dataset \
    --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
    --desc /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack/cifar100_aug/1_resnet18_clp_thres_thres_0.8/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4/

python eval-untarget.py --wb --data-dir /scratch/zx1673/dataset \
    --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
    --desc /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack/cifar10_aug/1_resnet18_clp_thres_thres_0.8/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4/






