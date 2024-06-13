#!/bin/sh

python PGDAT.py --Lambda 0.1 --gpu 1 --use_track_gradient --model WRN-28-10 --dataset cifar100
python PGDAT.py --Lambda 0.3 --gpu 1 --use_track_gradient --model WRN-28-10 --dataset cifar100
python PGDAT.py --Lambda 0.5 --gpu 1 --use_track_gradient --model WRN-28-10 --dataset cifar100
python PGDAT.py --Lambda 0.7 --gpu 1 --use_track_gradient --model WRN-28-10 --dataset cifar100
python PGDAT.py --Lambda 0.9 --gpu 1 --use_track_gradient --model WRN-28-10 --dataset cifar100
