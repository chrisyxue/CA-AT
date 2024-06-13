#!/bin/sh


python train_continual_regularization.py --dataset cifar100 --task_num 10 --Lambda 1 --epoch 200 --model ResNet34 --method LwF --gpu 3 --ewc_mode online
python train_continual_regularization.py --dataset cifar100 --task_num 5 --Lambda 1 --epoch 200 --model ResNet34 --method LwF --gpu 3 --ewc_mode online

python train_continual_regularization.py --dataset cifar100 --task_num 10 --Lambda 0.25 --epoch 200 --model ResNet34 --method LwF --gpu 3 --ewc_mode online
python train_continual_regularization.py --dataset cifar100 --task_num 5 --Lambda 0.25 --epoch 200 --model ResNet34 --method LwF --gpu 3 --ewc_mode online

python train_continual_regularization.py --dataset cifar100 --task_num 10 --Lambda 0.75 --epoch 200 --model ResNet34 --method LwF --gpu 3 --ewc_mode online
python train_continual_regularization.py --dataset cifar100 --task_num 5 --Lambda 0.75 --epoch 200 --model ResNet34 --method LwF --gpu 3 --ewc_mode online
