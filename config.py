import argparse



parser = argparse.ArgumentParser(description='PathFusion for MMEA')
parser.add_argument('--dataset', type=str, default='DB15K-FB15K', help='dataset name', choices=['DB15K-FB15K', 'YAGO15K-FB15K'])



args = parser.parse_args()

if args.dataset == 'db15k':
    from data.utils_db import *

else:
    from data.utils_yb import *

