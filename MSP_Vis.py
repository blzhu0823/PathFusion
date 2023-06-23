import argparse
import numpy as np
from tqdm import tqdm
from sinkhorn import matrix_sinkhorn
import sparse_eval
import torch



parser = argparse.ArgumentParser(description='MSP process (Vis) for PathFusion')
parser.add_argument('--dataset', type=str, default='DB15K-FB15K', help='dataset name', choices=['DB15K-FB15K', 'YAGO15K-FB15K'])
parser.add_argument('--max_image_num', type=int, default=6, help='max image num for each entity', choices=[1, 2, 3, 4, 5, 6])

args = parser.parse_args()


source_dataset, target_dataset = args.dataset.split('-')
source_dataset = source_dataset.lower()
target_dataset = target_dataset.lower()
use_img_num = args.max_image_num


if source_dataset == 'db15k':
    from data.utils_db import *
else:
    from data.utils_yb import *


# load source image embedding
with open('./data/image_embed/{}.npy'.format(source_dataset), 'rb') as f:
    source_embedding = pickle.load(f)

# load source entity id to image id mapping
with open('./data/image_embed/{}'.format(source_dataset), 'rb') as f:
    source_id2img = pickle.load(f)


print('source embedding shape: {}'.format(source_embedding.shape))
print('source id2img length: {}'.format(len(source_id2img)))


# load target image embedding
with open('./data/image_embed/{}.npy'.format(target_dataset), 'rb') as f:
    target_embedding = pickle.load(f)

# load target entity id to image id mapping
with open('./data/image_embed/{}'.format(target_dataset), 'rb') as f:
    target_id2img = pickle.load(f)


print('target embedding shape: {}'.format(target_embedding.shape))
print('target id2img length: {}'.format(len(target_id2img)))




print('source target entity num:', len(source2id), len(target2id))
# score init as -float('inf')
image_scores = np.zeros((len(source2id), len(target2id)))
image_scores = -float('inf') * np.ones((len(source2id), len(target2id)))


for i in tqdm(range(len(source2id))):
    for j in range(len(target2id)):
        for ii in range(min(use_img_num, len(source_id2img[i]))):
            for jj in range(min(use_img_num, len(target_id2img[j]))):
                image_scores[i, j] = max(image_scores[i, j], np.dot(source_embedding[source_id2img[i][ii]], target_embedding[target_id2img[j][jj]]))



dev_pair = np.array(dev_pair)
scores = np.zeros((len(dev_pair), len(dev_pair)), dtype=np.float32)
for i, l in enumerate(dev_pair[:, 0]):
    for j, r in enumerate(dev_pair[:, 1]):
        scores[i][j] = image_scores[l][r - len(source2id)] if image_scores[l][r - len(source2id)] != -float('inf') else 0


scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))


# save the scores as .npy file
np.save(f'./data/{args.dataset}/MSP_results/Vis.npy', scores)


# evaluate MSP result for Vis

scores = torch.Tensor(scores)
scores = matrix_sinkhorn(1 - scores)

sparse_eval.evaluate_sim_matrix(link = torch.stack([torch.arange(len(dev_pair)), 
                                        torch.arange(len(dev_pair))], dim=0),
                                        sim_x2y=scores,
                                        no_csls=True)