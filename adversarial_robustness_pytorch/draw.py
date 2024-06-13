import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
import pdb

# path_lst = [
#     # '/home/zhiyuxue/results/adver_robustness_notrack/cifar10_aug/1_resnet18_adver_avg_beta_0.0/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4',
#     '/home/zhiyuxue/results/adver_robustness_notrack/cifar10_aug/1_resnet18_adver_avg_beta_0.5/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4',
#     '/home/zhiyuxue/results/adver_robustness_notrack/cifar10_aug/1_resnet18_adver_avg_beta_1.0/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4',
#     '/home/zhiyuxue/results/adver_robustness_notrack/cifar10_aug/1_resnet18_adver_thres_thres_0.8/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4',
#     '/home/zhiyuxue/results/adver_robustness_notrack/cifar10_aug/1_resnet18_adver_thres_thres_0.9/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4'
# ]

# path_lst = [
#     # '/home/zhiyuxue/results/adver_robustness_notrack/cifar10_aug/1_resnet34_adver_avg_beta_0.0/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4',
#     '/home/zhiyuxue/results/adver_robustness_notrack/cifar10_aug/1_resnet34_adver_avg_beta_0.5/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4',
#     '/home/zhiyuxue/results/adver_robustness_notrack/cifar10_aug/1_resnet34_adver_avg_beta_1.0/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4',
#     '/home/zhiyuxue/results/adver_robustness_notrack/cifar10_aug/1_resnet34_adver_thres_thres_0.8/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4',
#     '/home/zhiyuxue/results/adver_robustness_notrack/cifar10_aug/1_resnet34_adver_thres_thres_0.9/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4'
# ]

# path_lst = [
#     # '/home/zhiyuxue/results/adver_robustness_notrack/cifar10/1_resnet34_adver_avg_beta_0.0/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4',
#     '/home/zhiyuxue/results/adver_robustness_notrack/cifar10/1_resnet34_adver_avg_beta_0.5/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4',
#     '/home/zhiyuxue/results/adver_robustness_notrack/cifar10/1_resnet34_adver_avg_beta_1.0/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4',
#     '/home/zhiyuxue/results/adver_robustness_notrack/cifar10/1_resnet34_adver_thres_thres_0.8/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4',
#     '/home/zhiyuxue/results/adver_robustness_notrack/cifar10/1_resnet34_adver_thres_thres_0.9/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4'
# ]
# title = 'CIFAR-10, ResNet34'

# path_lst = [
#     # '/home/zhiyuxue/results/adver_robustness_notrack/cifar10/1_resnet18_adver_avg_beta_0.0/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4',
#     '/home/zhiyuxue/results/adver_robustness_notrack/cifar10/1_resnet18_adver_avg_beta_0.5/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4',
#     '/home/zhiyuxue/results/adver_robustness_notrack/cifar10/1_resnet18_adver_avg_beta_1.0/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4',
#     '/home/zhiyuxue/results/adver_robustness_notrack/cifar10/1_resnet18_adver_thres_thres_0.8/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4',
#     '/home/zhiyuxue/results/adver_robustness_notrack/cifar10/1_resnet18_adver_thres_thres_0.9/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4'
# ]
# title = 'CIFAR-10, ResNet18'

# path_name_lst = [
#     # '$\lambda=0.0$',
#     '$\lambda=0.5$',
#     '$\lambda=1.0$',
#     'Ours,$\eta=0.8$',
#     'Ours,$\eta=0.9$'
# ]





path_lst = [
    '/home/zhiyuxue/results/adver_robustness_notrack/cifar10_adver_criterion_ce/1_resnet18_adver_avg_beta_0.5/linf-pgd_0.03137254901960784_0.00784313725490196_10/300_1024_0.4',
    '/home/zhiyuxue/results/adver_robustness_notrack/cifar10_adver_criterion_ce/1_resnet18_adver_avg_beta_1.0/linf-pgd_0.03137254901960784_0.00784313725490196_10/300_1024_0.4',
    '/home/zhiyuxue/results/adver_robustness_notrack/cifar10_adver_criterion_ce/1_resnet18_adver_thres_thres_0.8/linf-pgd_0.03137254901960784_0.00784313725490196_10/300_1024_0.4'
]
title = 'CIFAR-10, ResNet18, Batch Size 1024'
path_name_lst = [
    '$\lambda=0.5$',
    '$\lambda=1.0$',
    'Ours,$\eta=0.8$',
]



# path_lst = [
#     '/home/zhiyuxue/results/adver_robustness_notrack/cifar10_adver_criterion_ce/1_resnet18_adver_avg_beta_0.5/linf-pgd_0.03137254901960784_0.00784313725490196_10/300_512_0.4',
#     '/home/zhiyuxue/results/adver_robustness_notrack/cifar10_adver_criterion_ce/1_resnet18_adver_avg_beta_1.0/linf-pgd_0.03137254901960784_0.00784313725490196_10/300_512_0.4',
#     '/home/zhiyuxue/results/adver_robustness_notrack/cifar10_adver_criterion_ce/1_resnet18_adver_thres_thres_0.8/linf-pgd_0.03137254901960784_0.00784313725490196_10/300_512_0.4'
# ]
# title = 'CIFAR-10, ResNet18, Batch Size 512'
# path_name_lst = [
#     '$\lambda=0.5$',
#     '$\lambda=1.0$',
#     'Ours,$\eta=0.8$',
# ]

epoch_lst = [1,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300]


cos_dis_lst_all, adver_l2_lst_all, clean_l2_lst_all, tau_lst_all = [], [], [], []
for path in path_lst:
    cos_dis_lst, adver_l2_lst, clean_l2_lst, tau_lst = [], [], [], []
    for epoch in epoch_lst:
        csv_file = os.path.join(path, f'{epoch}_grad_track.csv')
        df = pd.read_csv(csv_file)
        cos_dis = 1 - df['Cos'].values[0]
        adver_l2 = df['Adver_L2Norm'].values[0]
        clean_l2 = df['Clean_L2Norm'].values[0]
        tau = cos_dis * adver_l2 * clean_l2
        cos_dis_lst.append(cos_dis)
        adver_l2_lst.append(adver_l2)
        clean_l2_lst.append(clean_l2)
        tau_lst.append(tau)
    cos_dis_lst_all.append(cos_dis_lst)
    adver_l2_lst_all.append(adver_l2_lst)
    clean_l2_lst_all.append(clean_l2_lst)
    tau_lst_all.append(tau_lst)




# Create a figure and a set of subplots with 1 row and 3 columns
fig, axes = plt.subplots(1, 4, figsize=(14, 3))  # You can adjust the size as needed
# First subplot for cos_dis_lst
for path_name, cos_dis_lst in zip(path_name_lst, cos_dis_lst_all):
    axes[0].plot(epoch_lst, cos_dis_lst, label=f'{path_name}', marker='o')
axes[0].set_ylabel('$\log [1 - cos(g_{a},g_{c})]$')
axes[0].set_yscale("log")
axes[0].grid()
axes[0].legend()
axes[0].set_xlabel('Epoch')

# Second subplot for adver_l2_lst
for path_name, adver_l2_lst in zip(path_name_lst, adver_l2_lst_all):
    axes[1].plot(epoch_lst, adver_l2_lst, label=f'{path_name}', marker='o')
axes[1].set_ylabel('$\log ||g_{a}||$')
axes[1].set_yscale("log")
axes[1].grid()
axes[1].legend()
axes[1].set_xlabel('Epoch')

# Third subplot for clean_l2_lst
for path_name, clean_l2_lst in zip(path_name_lst, clean_l2_lst_all):
    axes[2].plot(epoch_lst, clean_l2_lst, label=f'{path_name}', marker='o')
axes[2].set_ylabel('$\log ||g_{c}||$')
axes[2].set_yscale("log")
axes[2].grid()
axes[2].legend()
axes[2].set_xlabel('Epoch')

# Third subplot for clean_l2_lst
for path_name, tau_lst in zip(path_name_lst, tau_lst_all):
    axes[3].plot(epoch_lst, tau_lst, label=f'{path_name}', marker='o')
axes[3].set_ylabel('$\log \\tau$')
axes[3].set_yscale("log")
axes[3].grid()
axes[3].legend()
axes[3].set_xlabel('Epoch')

fig.suptitle(title)
# Adjust layout to prevent overlap
plt.tight_layout()

# Save the entire figure
plt.savefig('test_combined.png')


# plt.figure()
# for path_name, tau_lst in zip(path_name_lst, tau_lst_all):
#     plt.plot(epoch_lst, tau_lst, label=f'{path_name}',marker='o')
# plt.ylabel('$\tau$')

# plt.yscale("log")
# plt.grid()
# plt.legend()
# plt.xlabel('Epoch')
# plt.savefig('test.png')
# plt.plot(epoch_lst, cos_dis_lst, label='$1-Cos(g_a,g_c)$',marker='o')
# plt.plot(epoch_lst, adver_l2_lst, label='Adver_L2Norm',marker='o')
# plt.plot(epoch_lst, clean_l2_lst, label='Clean_L2Norm',marker='o')
# plt.grid()
# plt.xlabel('Epoch')
# plt.savefig('test.png')
# pdb.set_trace()
