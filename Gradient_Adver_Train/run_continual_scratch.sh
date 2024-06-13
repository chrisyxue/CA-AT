#!/bin/sh

# python train_continual.py --dataset cifar10 --task_num 2 --from_scratch --gpu 1 --epoch 100 --Lambda 0.5

python train_continual.py --dataset cifar100 --task_num 10 --from_scratch --gpu 1 --Lambda 0.5 --epoch 50 
python train_continual.py --dataset cifar100 --task_num 5 --from_scratch --gpu 1 --Lambda 0.5 --epoch 50

python train_continual.py --dataset cifar100 --task_num 10 --from_scratch --gpu 1 --Lambda 0 --epoch 50
python train_continual.py --dataset cifar100 --task_num 5 --from_scratch --gpu 1 --Lambda 0 --epoch 50