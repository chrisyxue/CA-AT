#!/bin/sh

python train_continual_regularization.py --dataset mnist --task_num 5 --Lambda 0  --epoch 10 --model SimpleCNN --method EWC --gpu 3 --decay cos --lr 0.01 --ewc_lambda 0
python train_continual_regularization.py --dataset mnist --task_num 5 --Lambda 0  --epoch 10 --model SimpleCNN --method EWC --gpu 3 --decay cos --lr 0.01 --ewc_lambda 0.01
python train_continual_regularization.py --dataset mnist --task_num 5 --Lambda 0  --epoch 10 --model SimpleCNN --method EWC --gpu 3 --decay cos --lr 0.01 --ewc_lambda 0.05
python train_continual_regularization.py --dataset mnist --task_num 5 --Lambda 0  --epoch 10 --model SimpleCNN --method EWC --gpu 3 --decay cos --lr 0.01 --ewc_lambda 0.1

python train_continual_regularization.py --dataset mnist --task_num 10 --Lambda 0  --epoch 10 --model SimpleCNN --method EWC --gpu 3 --decay cos --lr 0.01 --ewc_lambda 0
python train_continual_regularization.py --dataset mnist --task_num 10 --Lambda 0  --epoch 10 --model SimpleCNN --method EWC --gpu 3 --decay cos --lr 0.01 --ewc_lambda 0.01
python train_continual_regularization.py --dataset mnist --task_num 10 --Lambda 0  --epoch 10 --model SimpleCNN --method EWC --gpu 3 --decay cos --lr 0.01 --ewc_lambda 0.05
python train_continual_regularization.py --dataset mnist --task_num 10 --Lambda 0  --epoch 10 --model SimpleCNN --method EWC --gpu 3 --decay cos --lr 0.01 --ewc_lambda 0.1

python train_continual_regularization.py --dataset mnist --task_num 5 --Lambda 0  --epoch 10 --model SimpleCNN --method LwF --gpu 3 --decay cos --lr 0.01 --alpha 0
python train_continual_regularization.py --dataset mnist --task_num 5 --Lambda 0  --epoch 10 --model SimpleCNN --method LwF --gpu 3 --decay cos --lr 0.01 --alpha 0.01
python train_continual_regularization.py --dataset mnist --task_num 5 --Lambda 0  --epoch 10 --model SimpleCNN --method LwF --gpu 3 --decay cos --lr 0.01 --alpha 0.05
python train_continual_regularization.py --dataset mnist --task_num 5 --Lambda 0  --epoch 10 --model SimpleCNN --method LwF --gpu 3 --decay cos --lr 0.01 --alpha 0.1

python train_continual_regularization.py --dataset mnist --task_num 10 --Lambda 0  --epoch 10 --model SimpleCNN --method LwF --gpu 3 --decay cos --lr 0.01 --alpha 0
python train_continual_regularization.py --dataset mnist --task_num 10 --Lambda 0  --epoch 10 --model SimpleCNN --method LwF --gpu 3 --decay cos --lr 0.01 --alpha 0.01
python train_continual_regularization.py --dataset mnist --task_num 10 --Lambda 0  --epoch 10 --model SimpleCNN --method LwF --gpu 3 --decay cos --lr 0.01 --alpha 0.05
python train_continual_regularization.py --dataset mnist --task_num 10 --Lambda 0  --epoch 10 --model SimpleCNN --method LwF --gpu 3 --decay cos --lr 0.01 --alpha 0.1



python train_continual_regularization.py --dataset cifar100 --task_num 5 --Lambda 0  --epoch 70 --model SimpleCNN --method EWC --gpu 3 --decay cos --lr 0.1 --ewc_lambda 0
python train_continual_regularization.py --dataset cifar100 --task_num 5 --Lambda 0  --epoch 70 --model SimpleCNN --method EWC --gpu 3 --decay cos --lr 0.1 --ewc_lambda 0.01
python train_continual_regularization.py --dataset cifar100 --task_num 5 --Lambda 0  --epoch 70 --model SimpleCNN --method EWC --gpu 3 --decay cos --lr 0.1 --ewc_lambda 0.05
python train_continual_regularization.py --dataset cifar100 --task_num 5 --Lambda 0  --epoch 70 --model SimpleCNN --method EWC --gpu 3 --decay cos --lr 0.1 --ewc_lambda 0.1

python train_continual_regularization.py --dataset cifar100 --task_num 10 --Lambda 0  --epoch 70 --model SimpleCNN --method EWC --gpu 3 --decay cos --lr 0.1 --ewc_lambda 0
python train_continual_regularization.py --dataset cifar100 --task_num 10 --Lambda 0  --epoch 70 --model SimpleCNN --method EWC --gpu 3 --decay cos --lr 0.1 --ewc_lambda 0.01
python train_continual_regularization.py --dataset cifar100 --task_num 10 --Lambda 0  --epoch 70 --model SimpleCNN --method EWC --gpu 3 --decay cos --lr 0.1 --ewc_lambda 0.05
python train_continual_regularization.py --dataset cifar100 --task_num 10 --Lambda 0  --epoch 70 --model SimpleCNN --method EWC --gpu 3 --decay cos --lr 0.1 --ewc_lambda 0.1

python train_continual_regularization.py --dataset cifar100 --task_num 5 --Lambda 0  --epoch 70 --model SimpleCNN --method LwF --gpu 3 --decay cos --lr 0.1 --alpha 0
python train_continual_regularization.py --dataset cifar100 --task_num 5 --Lambda 0  --epoch 70 --model SimpleCNN --method LwF --gpu 3 --decay cos --lr 0.1 --alpha 0.01
python train_continual_regularization.py --dataset cifar100 --task_num 5 --Lambda 0  --epoch 70 --model SimpleCNN --method LwF --gpu 3 --decay cos --lr 0.1 --alpha 0.05
python train_continual_regularization.py --dataset cifar100 --task_num 5 --Lambda 0  --epoch 70 --model SimpleCNN --method LwF --gpu 3 --decay cos --lr 0.1 --alpha 0.1

python train_continual_regularization.py --dataset cifar100 --task_num 10 --Lambda 0  --epoch 70 --model SimpleCNN --method LwF --gpu 3 --decay cos --lr 0.1 --alpha 0
python train_continual_regularization.py --dataset cifar100 --task_num 10 --Lambda 0  --epoch 70 --model SimpleCNN --method LwF --gpu 3 --decay cos --lr 0.1 --alpha 0.01
python train_continual_regularization.py --dataset cifar100 --task_num 10 --Lambda 0  --epoch 70 --model SimpleCNN --method LwF --gpu 3 --decay cos --lr 0.1 --alpha 0.05
python train_continual_regularization.py --dataset cifar100 --task_num 10 --Lambda 0  --epoch 70 --model SimpleCNN --method LwF --gpu 3 --decay cos --lr 0.1 --alpha 0.1

# python train_continual_regularization.py --dataset cifar100 --task_num 10 --Lambda 0  --epoch 70 --model SimpleCNN --method Finetune --gpu 3
# python train_continual_regularization.py --dataset cifar100 --task_num 5 --Lambda 0  --epoch 70 --model SimpleCNN --method Finetune --gpu 3
# python train_continual_regularization.py --dataset cifar100 --task_num 10 --Lambda 0  --epoch 70 --model SimpleCNN --method EWC --gpu 3
# python train_continual_regularization.py --dataset cifar100 --task_num 5 --Lambda 0  --epoch 70 --model SimpleCNN --method EWC --gpu 3
# python train_continual_regularization.py --dataset cifar100 --task_num 10 --Lambda 0  --epoch 70 --model SimpleCNN --method EWC --gpu 3 --ewc_mode online
# python train_continual_regularization.py --dataset cifar100 --task_num 5 --Lambda 0  --epoch 70 --model SimpleCNN --method EWC --gpu 3 --ewc_mode online
# python train_continual_regularization.py --dataset cifar100 --task_num 10 --Lambda 0 --epoch 70 --model SimpleCNN --method LwF --gpu 3
# python train_continual_regularization.py --dataset cifar100 --task_num 5 --Lambda 0 --epoch 70 --model SimpleCNN --method LwF --gpu 3

# python train_continual_regularization.py --dataset cifar100 --task_num 20 --Lambda 0  --epoch 70 --model SimpleCNN --method Finetune --gpu 3
# python train_continual_regularization.py --dataset cifar100 --task_num 20 --Lambda 0  --epoch 70 --model SimpleCNN --method EWC --gpu 3
# python train_continual_regularization.py --dataset cifar100 --task_num 20 --Lambda 0  --epoch 70 --model SimpleCNN --method EWC --gpu 3 --ewc_mode online
# python train_continual_regularization.py --dataset cifar100 --task_num 20 --Lambda 0 --epoch 70 --model SimpleCNN --method LwF --gpu 3

# python train_continual_regularization.py --dataset cifar100 --task_num 20 --Lambda 0  --epoch 70 --model ResNet34 --method Finetune --gpu 3
# python train_continual_regularization.py --dataset cifar100 --task_num 20 --Lambda 0  --epoch 70 --model ResNet34 --method EWC --gpu 3
# python train_continual_regularization.py --dataset cifar100 --task_num 20 --Lambda 0  --epoch 70 --model ResNet34 --method EWC --gpu 3 --ewc_mode online
# python train_continual_regularization.py --dataset cifar100 --task_num 20 --Lambda 0 --epoch 70 --model ResNet34 --method LwF --gpu 3






# python train_continual_regularization.py --dataset cifar100 --task_num 10 --Lambda 0  --epoch 70 --model ResNet34 --method Finetune --gpu 3
# python train_continual_regularization.py --dataset cifar100 --task_num 5 --Lambda 0  --epoch 70 --model ResNet34 --method Finetune --gpu 3

# python train_continual_regularization.py --dataset cifar100 --task_num 10 --Lambda 0  --epoch 70 --model ResNet34 --method EWC --gpu 3
# python train_continual_regularization.py --dataset cifar100 --task_num 5 --Lambda 0  --epoch 70 --model ResNet34 --method EWC --gpu 3

# python train_continual_regularization.py --dataset cifar100 --task_num 10 --Lambda 0  --epoch 70 --model ResNet34 --method EWC --gpu 3 --ewc_mode online
# python train_continual_regularization.py --dataset cifar100 --task_num 5 --Lambda 0  --epoch 70 --model ResNet34 --method EWC --gpu 3 --ewc_mode online

# python train_continual_regularization.py --dataset cifar100 --task_num 10 --Lambda 0 --epoch 70 --model ResNet34 --method LwF --gpu 3
# python train_continual_regularization.py --dataset cifar100 --task_num 5 --Lambda 0 --epoch 70 --model ResNet34 --method LwF --gpu 3


