#!/bin/sh


python PGDAT.py --Lambda 0.1 --gpu 3 --use_track_gradient --dataset svhn
python PGDAT.py --Lambda 0.3 --gpu 3 --use_track_gradient --dataset svhn
python PGDAT.py --Lambda 0.5 --gpu 3 --use_track_gradient --dataset svhn
python PGDAT.py --Lambda 0.7 --gpu 3 --use_track_gradient --dataset svhn
python PGDAT.py --Lambda 0.9 --gpu 3 --use_track_gradient --dataset svhn
python PGDAT.py --Lambda 1 --gpu 3 --use_track_gradient --dataset svhn
