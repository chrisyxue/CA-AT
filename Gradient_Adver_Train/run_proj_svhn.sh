#!/bin/sh


python PGDAT.py --Lambda 0.1 --gpu 3 --dataset svhn
python PGDAT.py --Lambda 0.2 --gpu 3 --dataset svhn
python PGDAT.py --Lambda 0.3 --gpu 3 --dataset svhn
python PGDAT.py --Lambda 0.4 --gpu 3 --dataset svhn
python PGDAT.py --Lambda 0.6 --gpu 3 --dataset svhn
python PGDAT.py --Lambda 0.7 --gpu 3 --dataset svhn
python PGDAT.py --Lambda 0.8 --gpu 3 --dataset svhn
python PGDAT.py --Lambda 0.9 --gpu 3 --dataset svhn
