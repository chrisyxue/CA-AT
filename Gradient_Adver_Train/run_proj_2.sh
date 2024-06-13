#!/bin/sh

# python PGDAT_Proj.py --Lambda 0.1 --gpu 3 --grad_norm
# python PGDAT_Proj.py --Lambda 0.3 --gpu 3 --grad_norm
# python PGDAT_Proj.py --Lambda 0.5 --gpu 3 --grad_norm
# python PGDAT_Proj.py --Lambda 0.7 --gpu 3 --grad_norm
# python PGDAT.py --Lambda 0.2 --gpu 3
# python PGDAT_Proj.py --Lambda 0.2 --gpu 3 --grad_norm
# python PGDAT.py --Lambda 0.4 --gpu 3
# python PGDAT_Proj.py --Lambda 0.4 --gpu 3 --grad_norm
# python PGDAT.py --Lambda 0.6 --gpu 3
# python PGDAT_Proj.py --Lambda 0.6 --gpu 3 --grad_norm
# python PGDAT.py --Lambda 0.8 --gpu 3
# python PGDAT_Proj.py --Lambda 0.8 --gpu 3 --grad_norm
# python PGDAT.py --Lambda 0 --gpu 3
# python PGDAT_Proj.py --Lambda 0 --gpu 3 --grad_norm


# python PGDAT.py --Lambda 0 --gpu 3
# python PGDAT_Proj.py --Lambda 0 --gpu 3 --grad_norm

python PGDAT.py --Lambda 0.4 --gpu 2 --use_cat
python PGDAT_Proj.py --Lambda 0.4 --gpu 2 --grad_norm --use_cat
python PGDAT.py --Lambda 0.5 --gpu 2 --use_cat
python PGDAT_Proj.py --Lambda 0.5 --gpu 2 --grad_norm --use_cat
python PGDAT.py --Lambda 0.5 --gpu 2 --use_cat

python PGDAT_Proj.py --Lambda 0.7 --gpu 2 --grad_norm --use_cat
python PGDAT.py --Lambda 0.7 --gpu 2 --use_cat
python PGDAT_Proj.py --Lambda 0.5 --gpu 2 --grad_norm --use_cat
python PGDAT.py --Lambda 1 --gpu 2 --use_cat
python PGDAT_Proj.py --Lambda 1 --gpu 2 --grad_norm --use_cat