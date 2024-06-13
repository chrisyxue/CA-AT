#!/bin/sh


python train_continual_regularization.py --dataset cifar100 --task_num 10 --Lambda 0.25 --epoch 200 --model ResNet34 --method EWC --gpu 1
python train_continual_regularization.py --dataset cifar100 --task_num 5 --Lambda 0.25 --epoch 200 --model ResNet34 --method EWC --gpu 1

python train_continual_regularization.py --dataset cifar100 --task_num 10 --Lambda 0.75 --epoch 200 --model ResNet34 --method EWC --gpu 1 
python train_continual_regularization.py --dataset cifar100 --task_num 5 --Lambda 0.75 --epoch 200 --model ResNet34 --method EWC --gpu 1

python train_continual_regularization.py --dataset cifar100 --task_num 10 --Lambda 1 --epoch 200 --model ResNet34 --method EWC --gpu 1
python train_continual_regularization.py --dataset cifar100 --task_num 5 --Lambda 1 --epoch 200 --model ResNet34 --method EWC --gpu 1

