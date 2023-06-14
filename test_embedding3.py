import pickle
import numpy as np
from tqdm import tqdm
import os
import torch
import sparse_eval
from sinkhorn import matrix_sinkhorn
from os.path import join as pjoin




source = 'yago15k'
target = 'fb15k'
use_img_num = 6
print('fuck:', use_img_num)


if source == 'db15k':
    from data.utils_db import *
elif source == 'yago15k':
    from data.utils_yb import *



print()
print()
# load source image embedding
with open('./data/image_embed/{}.npy'.format(source), 'rb') as f:
    source_embedding = pickle.load(f)

# load source entity id to image id mapping
with open('./data/image_embed/{}'.format(source), 'rb') as f:
    source_id2img = pickle.load(f)


print('source embedding shape: {}'.format(source_embedding.shape))
print('source id2img length: {}'.format(len(source_id2img)))


# load target image embedding
with open('./data/image_embed/{}.npy'.format(target), 'rb') as f:
    target_embedding = pickle.load(f)

# load target entity id to image id mapping
with open('./data/image_embed/{}'.format(target), 'rb') as f:
    target_id2img = pickle.load(f)


print('target embedding shape: {}'.format(target_embedding.shape))
print('target id2img length: {}'.format(len(target_id2img)))



source_embedding_cnt = source_embedding.shape[0]
target_embedding_cnt = target_embedding.shape[0]

# construct source_id to source_embedding index mapping

source2embed = np.zeros((len(source2id), source_embedding_cnt))
for i in tqdm(range(len(source2id))):
    source2embed[i, source_id2img[i]] = 1

# construct target_id to target_embedding index mapping
target2embed = np.zeros((len(target2id), target_embedding_cnt))
for i in tqdm(range(len(target2id))):
    target2embed[i, target_id2img[i]] = 1


# load embedding 2 embedding score matrix

embed2embed = np.dot(source_embedding, target_embedding.T)



# get source2target score matrix
source2target = np.dot(source2embed, np.dot(embed2embed, target2embed.T))

# save source2target score matrix



dev_pair = np.array(dev_pair)
scores = np.zeros((len(dev_pair), len(dev_pair)), dtype=np.float32)
for i, l in enumerate(dev_pair[:, 0]):
    for j, r in enumerate(dev_pair[:, 1]):
        scores[i][j] = source2target[l][r - len(source2id)]


scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

with open('./data/image_embed/image_score{}_yago15k_sum_for_dev.npy'.format(use_img_num), 'wb') as f:
    pickle.dump(scores, f)

scores = torch.Tensor(scores)
scores = matrix_sinkhorn(1 - scores, device='cpu')
print(scores.shape)



sparse_eval.evaluate_sim_matrix(link = torch.stack([torch.arange(len(dev_pair)), 
                                        torch.arange(len(dev_pair))], dim=0),
                                        sim_x2y=torch.Tensor(scores),
                                        no_csls=True)