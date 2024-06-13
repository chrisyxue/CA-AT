#!/bin/sh

python train_gen_data.py --Lambda 0.6 --gpu 1
python train_gen_data.py --Lambda 0.7 --gpu 1
python train_gen_data.py --Lambda 0.8 --gpu 1
python train_gen_data.py --Lambda 0.9 --gpu 1
python train_gen_data.py --Lambda 1.0 --gpu 1