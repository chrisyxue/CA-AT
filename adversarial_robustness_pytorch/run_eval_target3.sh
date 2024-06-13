#!/bin/sh


# # Baseline
# python eval-target.py --wb --data-dir /scratch/zx1673/dataset \
#     --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
#     --desc /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack/cifar10/1_resnet18_trades_avg_beta_0.5/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4

# python eval-target.py --wb --data-dir /scratch/zx1673/dataset \
#     --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
#     --desc /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack/cifar10/1_resnet18_trades_avg_beta_1.0/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4

# python eval-target.py --wb --data-dir /scratch/zx1673/dataset \
#     --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
#     --desc /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack/cifar10/1_resnet34_trades_avg_beta_0.5/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4

# python eval-target.py --wb --data-dir /scratch/zx1673/dataset \
#     --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
#     --desc /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack/cifar10/1_resnet34_trades_avg_beta_1.0/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4

# python eval-target.py --wb --data-dir /scratch/zx1673/dataset \
#     --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
#     --desc /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack/cifar10/1_resnet50_trades_avg_beta_0.5/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4

# python eval-target.py --wb --data-dir /scratch/zx1673/dataset \
#     --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
#     --desc /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack/cifar10/1_resnet50_trades_avg_beta_1.0/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4


# # Grad Surgey
# python eval-target.py --wb --data-dir /scratch/zx1673/dataset \
#     --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
#     --desc /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack/cifar10/1_resnet18_trades_thres_thres_0.8/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4

# python eval-target.py --wb --data-dir /scratch/zx1673/dataset \
#     --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
#     --desc /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack/cifar10/1_resnet18_trades_thres_thres_0.9/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4

# python eval-target.py --wb --data-dir /scratch/zx1673/dataset \
#     --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
#     --desc /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack/cifar10/1_resnet34_trades_thres_thres_0.8/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4

# python eval-target.py --wb --data-dir /scratch/zx1673/dataset \
#     --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
#     --desc /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack/cifar10/1_resnet34_trades_thres_thres_0.9/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4

# python eval-target.py --wb --data-dir /scratch/zx1673/dataset \
#     --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
#     --desc /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack/cifar100_aug/1_resnet50_trades_thres_thres_0.8/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4

# python eval-target.py --wb --data-dir /scratch/zx1673/dataset \
#     --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
#     --desc /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack/cifar100_aug/1_resnet50_trades_thres_thres_0.9/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4




# python eval-target.py --wb --data-dir /scratch/zx1673/dataset \
#     --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
#     --desc /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack/cifar100_aug/1_resnet34_adver_thres_thres_0.8_ls_0.1/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4

# python eval-target.py --wb --data-dir /scratch/zx1673/dataset \
#     --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
#     --desc /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack/cifar100_aug/1_resnet34_adver_thres_thres_0.9_ls_0.1/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4

python eval-target.py --wb --data-dir /scratch/zx1673/dataset \
    --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
    --desc /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack/tiny-imagenet/1_preact-resnet18_adver_thres_thres_0.9/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4/





