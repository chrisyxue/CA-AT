#!/bin/sh



# CUDA_VISIBLE_DEVICES=4 nohup python eval-aa.py --wb --data-dir /data/zhiyu/dataset \
#     --log-dir /data/zhiyu/results/xuezhiyu/adver_robustness_notrack \
#     --desc /data/zhiyu/results/xuezhiyu/adver_robustness_notrack/cifar10_aug_adver_criterion_ce/1_wrn-28-10_clp_avg_beta_0.5/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4 &

# CUDA_VISIBLE_DEVICES=4 nohup python eval-aa.py --wb --data-dir /data/zhiyu/dataset \
#     --log-dir /data/zhiyu/results/xuezhiyu/adver_robustness_notrack \
#     --desc /data/zhiyu/results/xuezhiyu/adver_robustness_notrack/cifar10_aug_adver_criterion_ce/1_wrn-28-10_clp_avg_beta_0.75/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4 &



# CUDA_VISIBLE_DEVICES=4 nohup python eval-aa.py --wb --data-dir /data/zhiyu/dataset \
#     --log-dir /data/zhiyu/results/xuezhiyu/adver_robustness_notrack \
#     --desc /data/zhiyu/results/xuezhiyu/adver_robustness_notrack/cifar10_aug_adver_criterion_ce/1_wrn-28-10_clp_thres_thres_0.7/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4 &

CUDA_VISIBLE_DEVICES=9 nohup python eval-aa.py --wb --data-dir /data/zhiyu/dataset \
    --log-dir /data/zhiyu/results/xuezhiyu/adver_robustness_notrack \
    --desc /data/zhiyu/results/xuezhiyu/adver_robustness_notrack/cifar10_aug_adver_criterion_ce/1_wrn-28-10_clp_thres_thres_0.75/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4 &
CUDA_VISIBLE_DEVICES=9 nohup python eval-aa.py --wb --data-dir /data/zhiyu/dataset \
    --log-dir /data/zhiyu/results/xuezhiyu/adver_robustness_notrack \
    --desc /data/zhiyu/results/xuezhiyu/adver_robustness_notrack/cifar10_aug_adver_criterion_ce/1_wrn-28-10_clp_thres_thres_0.8/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4 &
CUDA_VISIBLE_DEVICES=9 nohup python eval-aa.py --wb --data-dir /data/zhiyu/dataset \
    --log-dir /data/zhiyu/results/xuezhiyu/adver_robustness_notrack \
    --desc /data/zhiyu/results/xuezhiyu/adver_robustness_notrack/cifar10_aug_adver_criterion_ce/1_wrn-28-10_clp_thres_thres_0.85/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4 &
CUDA_VISIBLE_DEVICES=9 nohup python eval-aa.py --wb --data-dir /data/zhiyu/dataset \
    --log-dir /data/zhiyu/results/xuezhiyu/adver_robustness_notrack \
    --desc /data/zhiyu/results/xuezhiyu/adver_robustness_notrack/cifar10_aug_adver_criterion_ce/1_wrn-28-10_clp_thres_thres_0.9/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4 &