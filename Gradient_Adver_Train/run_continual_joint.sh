#!/bin/sh

python train_continual_joint.py --dataset cifar100 --task_num 10 --Lambda 0.5 --epoch 240 --model ResNet34
python train_continual_joint.py --dataset cifar100 --task_num 5 --Lambda 0.5 --epoch 240 --model ResNet34

python train_continual_joint.py --dataset cifar100 --task_num 10 --Lambda 0.5 --epoch 100 --model SimpleCNN
python train_continual_joint.py --dataset cifar100 --task_num 5 --Lambda 0.5 --epoch 100 --model SimpleCNN
