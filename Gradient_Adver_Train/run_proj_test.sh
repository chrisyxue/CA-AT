#!/bin/sh

# python PGDAT_Proj.py --Lambda 0.1 --gpu 3 --grad_norm
# python PGDAT_Proj.py --Lambda 0.3 --gpu 3 --grad_norm
# python PGDAT_Proj.py --Lambda 0.5 --gpu 3 --grad_norm
# python PGDAT_Proj.py --Lambda 0.7 --gpu 3 --grad_norm
# python PGDAT.py --Lambda 0.2 --gpu 3
# python PGDAT_Proj.py --Lambda 0.2 --gpu 3 --grad_norm
# python PGDAT.py --Lambda 0.4 --gpu 3
# python PGDAT_Proj.py --Lambda 0.4 --gpu 3 --grad_norm
# python PGDAT.py --Lambda 0.6 --gpu 3
# python PGDAT_Proj.py --Lambda 0.6 --gpu 3 --grad_norm
# python PGDAT.py --Lambda 0.8 --gpu 3
# python PGDAT_Proj.py --Lambda 0.8 --gpu 3 --grad_norm
# python PGDAT.py --Lambda 0 --gpu 3
# python PGDAT_Proj.py --Lambda 0 --gpu 3 --grad_norm


# python PGDAT_Proj_test.py --Lambda 0.1 --gpu 3 --grad_norm
# python PGDAT_Proj_test.py --Lambda 0.2 --gpu 3 --grad_norm
# python PGDAT_Proj_test.py --Lambda 0.3 --gpu 3 --grad_norm
# python PGDAT_Proj_test.py --Lambda 0.4 --gpu 3 --grad_norm
# python PGDAT_Proj_test.py --Lambda 0.5 --gpu 3 --grad_norm
# python PGDAT_Proj_test.py --Lambda 0.6 --gpu 3 --grad_norm
# python PGDAT_Proj_test.py --Lambda 0.7 --gpu 3 --grad_norm
# python PGDAT_Proj_test.py --Lambda 0.8 --gpu 3 --grad_norm
# python PGDAT_Proj_test.py --Lambda 0.9 --gpu 3 --grad_norm

# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-8-7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.5

# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-grad-8-7-proj_thres0.9_NoGradNorm__e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --steps 20
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-grad-8-7-proj_thres0.8_NoGradNorm__e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --steps 20
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-grad-8-7-proj_thres0.5_NoGradNorm__e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --steps 20

# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-8-7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.0 --steps 20
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-8-7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --steps 20
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-8-7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.2 --steps 20
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-8-7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.3 --steps 20
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-8-7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.4 --steps 20
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-8-7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.5 --steps 20
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-8-7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.6 --steps 20
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-8-7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.7 --steps 20
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-8-7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.8 --steps 20
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-8-7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.9 --steps 20
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-8-7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda1.0 --steps 20

# /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-grad-8-7-proj_thres0.9_NoGradNorm_-TRADES_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-grad-8-7-proj_thres0.9-TRADES_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --steps 7
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-grad-8-7-proj_thres0.8-TRADES_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --steps 7
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-grad-8-7-proj_thres0.5-TRADES_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --steps 7

# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-grad-8-7-proj_thres0.9_NoGradNorm_-TRADES_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --steps 7


# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-grad-8-7-proj_thres0.9_NoGradNorm__e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --steps 7
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-grad-8-7-proj_thres0.8_NoGradNorm__e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --steps 7
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-grad-8-7-proj_thres0.5_NoGradNorm__e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --steps 7

# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-8-7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.0 --steps 7
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-8-7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --steps 7
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-8-7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.2 --steps 7
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-8-7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.3 --steps 7
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-8-7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.4 --steps 7
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-8-7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.5 --steps 7
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-8-7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.6 --steps 7
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-8-7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.7 --steps 7
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-8-7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.8 --steps 7
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-8-7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.9 --steps 7
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-8-7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda1.0 --steps 7


# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-grad-8-7-proj_thres0.9_NoGradNorm__e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --steps 7
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-grad-8-7-proj_thres0.8_NoGradNorm__e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --steps 7
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-grad-8-7-proj_thres0.5_NoGradNorm__e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --steps 7



# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-grad-8-20-proj_thres0.9_NoGradNorm__e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --steps 7
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-grad-8-15-proj_thres0.9_NoGradNorm__e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --steps 7
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-grad-8-10-proj_thres0.9_NoGradNorm__e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --steps 7

# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-grad-8-20-proj_thres0.95_NoGradNorm_-Pretrain-_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --steps 20
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-grad-8-15-proj_thres0.95_NoGradNorm_-Pretrain-_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --steps 20
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-grad-8-10-proj_thres0.9_NoGradNorm_-Pretrain-_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --steps 20
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34/untargeted-pgd-grad-8-7-proj_thres0.9_NoGradNorm_-Pretrain-_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --steps 20

# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/WRN-28-10/untargeted-pgd-8-7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --steps 20 --model WRN-28-10
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/WRN-28-10/untargeted-pgd-8-7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.3 --steps 20 --model WRN-28-10
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/WRN-28-10/untargeted-pgd-8-7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.5 --steps 20 --model WRN-28-10
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/WRN-28-10/untargeted-pgd-8-7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.7 --steps 20 --model WRN-28-10
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/WRN-28-10/untargeted-pgd-8-7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.9 --steps 20 --model WRN-28-10


# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/WRN-28-10/untargeted-pgd-grad-8-7-proj_thres0.9_NoGradNorm__e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --steps 20 --model WRN-28-10

# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34_old/ResNet34/untargeted-pgd-grad-8-7-proj_thres0.9_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --steps 20 --model resnet34
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34_old/ResNet34/untargeted-pgd-grad-8-7-proj_thres0.8_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --steps 20 --model resnet34
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34_old/ResNet34/untargeted-pgd-grad-8-7-proj_thres0.7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --steps 20 --model resnet34
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/ResNet34_old/ResNet34/untargeted-pgd-grad-8-7-proj_thres0.6_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --steps 20 --model resnet34

# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/WRN-28-10/untargeted-pgd-grad-8-7-proj_thres0.9_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --steps 20 --model WRN-28-10
# python PGDAT_Proj_test.py --save_folder /localscratch2/xuezhiyu/Gradient_Adver_Train/cifar10/WRN-28-10/untargeted-pgd-grad-8-7-proj_thres0.95_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --steps 20 --model WRN-28-10

# CUDA_VISIBLE_DEVICES=1 python PGDAT_Proj_test.py --save_folde /root/results/xuezhiyu/Gradient_Adver_Train/cifar100/resnet50/untargeted-pgd-grad-8-7-proj_thres0.8_e200-b128_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --model resnet50 --dataset cifar100 --steps 20
# CUDA_VISIBLE_DEVICES=1 python PGDAT_Proj_test.py --save_folde /root/results/xuezhiyu/Gradient_Adver_Train/cifar100/resnet50/untargeted-pgd-grad-8-7-proj_thres0.8_NoGradNorm__e200-b128_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --model resnet50 --dataset cifar100 --steps 20

# CUDA_VISIBLE_DEVICES=1 python PGDAT_Proj_test.py --save_folde /root/results/xuezhiyu/Gradient_Adver_Train/cifar100/resnet50/untargeted-pgd-grad-8-7-proj_thres0.7_e200-b128_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --model resnet50 --dataset cifar100 --steps 20
# CUDA_VISIBLE_DEVICES=1 python PGDAT_Proj_test.py --save_folde /root/results/xuezhiyu/Gradient_Adver_Train/cifar100/resnet50/untargeted-pgd-grad-8-7-proj_thres0.7_NoGradNorm__e200-b128_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --model resnet50 --dataset cifar100 --steps 20
CUDA_VISIBLE_DEVICES=1 python PGDAT_Proj_test.py --model WRN-28-10 --save_folder /root/results/xuezhiyu/Gradient_Adver_Train/cifar100/WRN-28-10/untargeted-pgd-8-7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.9 --dataset cifar100 --steps 20

CUDA_VISIBLE_DEVICES=1 python PGDAT_Proj_test.py --model WRN-28-10 --save_folder /root/results/xuezhiyu/Gradient_Adver_Train/cifar100/WRN-28-10/untargeted-pgd-grad-8-7-proj_thres0.6_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --dataset cifar100 --steps 20
CUDA_VISIBLE_DEVICES=1 python PGDAT_Proj_test.py --model WRN-28-10 --save_folder /root/results/xuezhiyu/Gradient_Adver_Train/cifar100/WRN-28-10/untargeted-pgd-grad-8-7-proj_thres0.7_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --dataset cifar100 --steps 20
CUDA_VISIBLE_DEVICES=1 python PGDAT_Proj_test.py --model WRN-28-10 --save_folder /root/results/xuezhiyu/Gradient_Adver_Train/cifar100/WRN-28-10/untargeted-pgd-grad-8-7-proj_thres0.8_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --dataset cifar100 --steps 20
CUDA_VISIBLE_DEVICES=1 python PGDAT_Proj_test.py --model WRN-28-10 --save_folder /root/results/xuezhiyu/Gradient_Adver_Train/cifar100/WRN-28-10/untargeted-pgd-grad-8-7-proj_thres0.9_e200-b256_sgd-lr0.1-m0.9-wd0.0005_cos_lambda0.1 --dataset cifar100 --steps 20
