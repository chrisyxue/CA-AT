#!/bin/sh
# python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.2 --use_trades
# python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.3 --use_trades
# python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.4 --use_trades
# # python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.5
# python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.6 --use_trades
# python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.7 --use_trades
# # python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.8
# python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.9 --use_trades

# python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.9 --steps 20
# python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.8 --steps 20
# python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.7 --steps 20

# python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.9 --steps 10
# python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.8 --steps 10
# python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.7 --steps 10

# python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.9 --steps 15
# python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.8 --steps 15
# python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.7 --steps 15


python PGDAT_Proj.py --Lambda 0.1 --gpu 1 --grad_proj thres --grad_thres 0.9 --use_cat --use_trades 
python PGDAT_Proj.py --Lambda 0.1 --gpu 1 --grad_proj thres --grad_thres 0.95 --use_cat --use_trades
python PGDAT_Proj.py --Lambda 0.1 --gpu 1 --grad_proj thres --grad_thres 0.9 --grad_norm --use_cat --use_trades
python PGDAT_Proj.py --Lambda 0.1 --gpu 1 --grad_proj thres --grad_thres 0.95 --grad_norm --use_cat --use_trades


# python PGDAT_Proj.py --Lambda 0.1 --gpu 2 --grad_proj thres --grad_thres 0.9 --steps 20 --use_pretrain