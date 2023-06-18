import argparse
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm, trange
from sinkhorn import matrix_sinkhorn
import sparse_eval


parser = argparse.ArgumentParser(description='MSP process (Attr) for PathFusion')
parser.add_argument('--dataset', type=str, default='DB15K-FB15K', help='dataset name', choices=['DB15K-FB15K', 'YAGO15K-FB15K'])

args = parser.parse_args()

source_dataset, target_dataset = args.dataset.split('-')



if source_dataset == 'DB15K':
    from data.utils_db import *
else:
    from data.utils_yb import *


source_keyValue_sents = sorted(list(source_attr_value_set))
target_keyValue_sents = sorted(list(target_attr_value_set))


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model = SentenceTransformer('Maite89/Roberta_finetuning_semantic_similarity_stsb_multi_mt').to(device)



source_key_embeddings = []
target_key_embeddings = []
source_value = []
target_value = []

batch_size = 128
for i in trange(0, len(source_keyValue_sents), batch_size):
    key_sents = source_keyValue_sents[i:i + batch_size]
    for j in range(len(key_sents)):
        try:
            source_value.append(float(key_sents[j].split(' ')[1]))
        except:
            source_value.append(0)
        key_sents[j] = key_sents[j].split(' ')[0]
    source_key_embeddings.append(model.encode(key_sents))
source_key_embeddings = np.concatenate(source_key_embeddings, axis=0)

for i in tqdm(range(0, len(target_keyValue_sents), batch_size)):
    key_sents = target_keyValue_sents[i:i + batch_size]
    for j in range(len(key_sents)):
        try:
            target_value.append(float(key_sents[j].split(' ')[1]))
        except:
            target_value.append(0)
        key_sents[j] = key_sents[j].split(' ')[0]
    target_key_embeddings.append(model.encode(target_keyValue_sents[i:i + batch_size]))
target_key_embeddings = np.concatenate(target_key_embeddings, axis=0)

source_value = np.array(source_value)[:, np.newaxis]
target_value = np.array(target_value)[np.newaxis, :]
scores_key = np.matmul(source_key_embeddings, target_key_embeddings.T)
scores_value = 1 / (np.abs(source_value - target_value) + 1e-3)

attr2attr = scores_key * scores_value

source2target = source2attr @ attr2attr @ target2attr.T

scores = np.zeros((len(dev_pair), len(dev_pair)), dtype=np.float32)
for i in range(len(dev_pair)):
    for j in range(len(dev_pair)):
        scores[i][j] = source2target[dev_pair[i][0]][dev_pair[j][1] - len(source2id)]


scores = (scores - scores.min()) / (scores.max() - scores.min())

# save the scores as .npy file
np.save(f'./data/{args.dataset}/MSP_results/Attr.npy', scores)


# evaluate MSP result for Attr

scores = torch.Tensor(scores)
scores = matrix_sinkhorn(1 - scores)


sparse_eval.evaluate_sim_matrix(link = torch.stack([torch.arange(len(dev_pair)), 
                                        torch.arange(len(dev_pair))], dim=0),
                                        sim_x2y=scores,
                                        no_csls=True)