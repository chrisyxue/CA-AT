#!/bin/sh


CUDA_VISIBLE_DEVICES=0 python eval-pgd.py --wb --data-dir <your-data-dir> \
    --log-dir <your-log-dir> \
    --desc checkpoints/1_resnet18_adver_avg_beta_0.5/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4

CUDA_VISIBLE_DEVICES=0 python eval-pgd.py --wb --data-dir <your-data-dir> \
    --log-dir <your-log-dir> \
    --desc checkpoints/1_resnet18_adver_thres_thres_0.8/linf-pgd_0.03137254901960784_0.00784313725490196_10/200_1024_0.4