import argparse



parser = argparse.ArgumentParser(description='PathFusion for MMEA')
parser.add_argument('--dataset', type=str, default='ICEWS05-15', help='dataset name', choices=['ICEWS05-15', 'YAGO-WIKI50K'])
parser.add_argument('--train_ratio', type=float, default=1, help='ratio of training seeds')



args = parser.parse_args()
