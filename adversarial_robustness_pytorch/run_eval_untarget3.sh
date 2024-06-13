#!/bin/sh


# # Baseline
python eval-target.py --wb --data-dir /scratch/zx1673/dataset \
    --log-dir /scratch/zx1673/results/xuezhiyu/adver_robustness_notrack \
    --desc 

/data/zhiyu/results/xuezhiyu/adver_robustness_notrack/cifar10_aug_adver_criterion_ce/1_wrn-28-10_clp_avg_beta_0.25/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4
/data/zhiyu/results/xuezhiyu/adver_robustness_notrack/cifar10_aug_adver_criterion_ce/1_wrn-28-10_clp_avg_beta_0.75/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4

