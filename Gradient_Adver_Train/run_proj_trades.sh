#!/bin/sh

python PGDAT_Proj.py --use_trades --gpu 1 --Lambda 0.1 --grad_norm
python PGDAT_Proj.py --use_trades --gpu 1 --Lambda 0.3 --grad_norm
python PGDAT_Proj.py --use_trades --gpu 1 --Lambda 0.5 --grad_norm
python PGDAT_Proj.py --use_trades --gpu 1 --Lambda 0.7 --grad_norm
python PGDAT_Proj.py --use_trades --gpu 1 --Lambda 0.9 --grad_norm