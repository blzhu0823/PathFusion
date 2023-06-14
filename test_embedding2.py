from data.utils_db import *
import pickle
import torch
import sparse_eval
from sinkhorn import matrix_sinkhorn
from test_rule import attr_signal_pair
from os.path import join as pjoin
from tqdm import tqdm


use_num = 6

print('fuck:', use_num)


with open('./data/image_embed/image_score{}_db15k_sum.npy'.format(use_num), 'rb') as f:
    image_scores = pickle.load(f)


dev_pair = np.array(dev_pair)
scores = np.zeros((len(dev_pair), len(dev_pair)), dtype=np.float32)
for i, l in enumerate(dev_pair[:, 0]):
    for j, r in enumerate(dev_pair[:, 1]):
        scores[i][j] = image_scores[l][r - len(source2id)] if image_scores[l][r - len(source2id)] != -float('inf') else 0


scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

# # row max min normalization
# for i in range(len(scores)):
#     scores[i] = (scores[i] - np.min(scores[i])) / (np.max(scores[i]) - np.min(scores[i]))
# save scores
with open('./data/image_embed/image_score{}_db15k_sum_for_dev.npy'.format(use_num), 'wb') as f:
    pickle.dump(scores, f)
    
scores = torch.Tensor(scores)
scores = matrix_sinkhorn(1 - scores, device='cpu')
print(scores.shape)



sparse_eval.evaluate_sim_matrix(link = torch.stack([torch.arange(len(dev_pair)), 
                                        torch.arange(len(dev_pair))], dim=0),
                                        sim_x2y=torch.Tensor(scores),
                                        no_csls=True)