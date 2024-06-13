#!/bin/sh

python train_gen_data.py --Lambda 0.1
python train_gen_data.py --Lambda 0.2
python train_gen_data.py --Lambda 0.3
python train_gen_data.py --Lambda 0.4
python train_gen_data.py --Lambda 0.5