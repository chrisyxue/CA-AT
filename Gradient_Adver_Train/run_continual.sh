#!/bin/sh


python train_continual.py --dataset cifar100 --task_num 10 --Lambda 0.5 --epoch 20 --model SimpleCNN
python train_continual.py --dataset cifar100 --task_num 5 --Lambda 0.5 --epoch 20 --model SimpleCNN
