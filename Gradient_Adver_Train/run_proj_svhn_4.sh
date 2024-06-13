#!/bin/sh

python PGDAT.py --Lambda 0.1 --gpu 1 --dataset svhn --model resnet18
python PGDAT.py --Lambda 0.2 --gpu 1 --dataset svhn --model resnet18
python PGDAT.py --Lambda 0.3 --gpu 1 --dataset svhn --model resnet18
python PGDAT.py --Lambda 0.4 --gpu 1 --dataset svhn --model resnet18
python PGDAT.py --Lambda 0.6 --gpu 1 --dataset svhn --model resnet18
python PGDAT.py --Lambda 0.7 --gpu 1 --dataset svhn --model resnet18
python PGDAT.py --Lambda 0.8 --gpu 1 --dataset svhn --model resnet18
python PGDAT.py --Lambda 0.9 --gpu 1 --dataset svhn --model resnet18
