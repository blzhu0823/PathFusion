from data.utils_yb import *
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from torch_kmeans import KMeans
from collections import defaultdict, Counter



attr_signal_pair = Counter()
attr_signal_diffsum = defaultdict(float)
for source_id, target_id in train_pair:
    source_attrs = id2attrs[source_id]
    target_attrs = id2attrs[target_id]
    min_diff_v = float('inf')
    min_diff_attr_pair = None
    for source_attr in source_attrs:
        source_key, source_value, source_schema = source_attr
        source_value = date2float(source_value)
        for target_attr in target_attrs:
            target_key, target_value = target_attr
            target_value = date2float(target_value)
            try:
                if abs(float(source_value) - float(target_value)) < min_diff_v:
                    min_diff_v = abs(float(source_value) - float(target_value))
                    min_diff_attr_pair = (source_key, target_key)
            except:
                pass
    attr_signal_pair[min_diff_attr_pair] += 1
    attr_signal_diffsum[min_diff_attr_pair] += min_diff_v



attr_signal_mean_diff = defaultdict(float)
for attr_pair, count in attr_signal_pair.items():
    attr_signal_mean_diff[attr_pair] = attr_signal_diffsum[attr_pair] / count

attr_signal_mean_diff = sorted(attr_signal_mean_diff.items(), key=lambda x: x[1], reverse=False)
with open('data/YAGO15K-FB15K/rule_attr', 'w') as f:
    for attr_pair, meandiff in attr_signal_mean_diff:
        f.write(str(attr_pair) + '\t' + str(meandiff) + '\t' + str(attr_signal_pair[attr_pair]))
        f.write('\n')