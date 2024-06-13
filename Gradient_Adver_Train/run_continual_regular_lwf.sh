#!/bin/sh


python train_continual_regularization.py --dataset cifar100 --task_num 10 --Lambda 0.5 --epoch 200 --model ResNet34 --method LwF --gpu 2
python train_continual_regularization.py --dataset cifar100 --task_num 5 --Lambda 0.5 --epoch 200 --model ResNet34 --method LwF --gpu 2

python train_continual_regularization.py --dataset cifar100 --task_num 10 --Lambda 0.5 --epoch 200 --model ResNet34 --method Finetune --gpu 2
python train_continual_regularization.py --dataset cifar100 --task_num 5 --Lambda 0.5 --epoch 200 --model ResNet34 --method Finetune --gpu 2
