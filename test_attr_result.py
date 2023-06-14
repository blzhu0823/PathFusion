from data.utils_db import *
import pickle
import torch
import sparse_eval
from sinkhorn import matrix_sinkhorn
from test_rule import attr_signal_mean_diff
from os.path import join as pjoin
from tqdm import tqdm
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# gnn_scores = np.load(pjoin('data', 'DB15K-FB15K', 'gnn_scores.pkl'), allow_pickle=True)
# source2attr = np.load(pjoin('data', 'DB15K-FB15K', 'source2attr.npy'), allow_pickle=True)
# target2attr = np.load(pjoin('data', 'DB15K-FB15K', 'target2attr.npy'), allow_pickle=True)
# attr2attr = np.load(pjoin('data', 'DB15K-FB15K', 'tmp.pkl'), allow_pickle=True)


# # # filter attr2attr
# # for i in range(len(attr2attr)):
# #     for j in range(len(attr2attr[i])):
# #         source_key = source_id2_attr_value[i].split(' ')[0]
# #         target_key = target_id2_attr_value[j].split(' ')[0]
# #         if (source_key, target_key) not in attr_signal_mean_diff:
# #             attr2attr[i][j] = 0




# source2target = source2attr @ attr2attr @ target2attr.T
# scores = np.zeros((len(dev_pair), len(dev_pair)), dtype=np.float32)
# for i in range(len(dev_pair)):
#     for j in range(len(dev_pair)):
#         scores[i][j] = source2target[dev_pair[i][0]][dev_pair[j][1] - len(source2id)]
# # save scores
# pickle.dump(scores, open(pjoin('data', 'DB15K-FB15K', 'tmptmp.pkl'), 'wb'))












# ----------------- test for saved pickle -----------------
# load image embeddings
# image_embeddings = np.load(pjoin('data', 'DB15K-FB15K', 'concat_embedding.npy'), allow_pickle=True)

# # calculate image scores
# image_scores = np.zeros((len(dev_pair), len(dev_pair)), dtype=np.float32)
# for i in tqdm(range(len(dev_pair))):
#     for j in range(len(dev_pair)):
#         image_scores[i][j] = np.dot(image_embeddings[dev_pair[i][0]], image_embeddings[dev_pair[j][1]])

# # save image scores
# pickle.dump(image_scores, open(pjoin('data', 'DB15K-FB15K', 'image_scores.pkl'), 'wb'))

# load image scores
# image_scores = np.load(pjoin('data', 'DB15K-FB15K', 'image_scores.pkl'), allow_pickle=True)
# norm image scores
# image_scores = (image_scores - np.min(image_scores)) / (np.max(image_scores) - np.min(image_scores))

# with open(pjoin('data', 'image_embed', 'image_score{}_db15k_max_for_dev.npy'.format(6)), 'rb') as f:
#     image_scores = pickle.load(f)

# load attr scores
attr_scores = np.load(pjoin('data', 'DB15K-FB15K', 'tmptmp.pkl'), allow_pickle=True)
# norm attr scores
attr_scores = (attr_scores - np.min(attr_scores)) / (np.max(attr_scores) - np.min(attr_scores))


total_scores = torch.Tensor(attr_scores)
# total_scores = torch.Tensor(attr_scores)
total_scores = matrix_sinkhorn(1 - total_scores)
print(total_scores.shape)



sparse_eval.evaluate_sim_matrix(link = torch.stack([torch.arange(len(dev_pair)), 
                                        torch.arange(len(dev_pair))], dim=0),
                                        sim_x2y=total_scores,
                                        no_csls=True)
