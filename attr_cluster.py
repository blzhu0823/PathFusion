from data.utils_db import *
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from torch_kmeans import KMeans
from collections import defaultdict

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# model = SentenceTransformer('Maite89/Roberta_finetuning_semantic_similarity_stsb_multi_mt').to(device)

source_keys_list = list(source_keys)
target_keys_list = list(target_keys)
key2seeds_id = defaultdict(list)


# deal with seeds feature for each key (in train_pair)
for i in range(len(train_pair)):
    source_entity_id, target_entity_id = train_pair[i]
    attr_value_schemas = id2attrs[source_entity_id]
    for attr, value, schema in attr_value_schemas:
        key2seeds_id[attr].append(i)

# get seeds one hot feature for each key
source_seed_feature = np.zeros((len(source_keys), len(train_pair)))
target_seed_feature = np.zeros((len(target_keys), len(train_pair)))
for i in range(len(source_keys_list)):
    for j in key2seeds_id[source_keys_list[i]]:
        source_seed_feature[i][j] = 1.0

for i in range(len(target_keys_list)):
    for j in key2seeds_id[target_keys_list[i]]:
        target_seed_feature[i][j] = 1.0

source_seed_feature = torch.tensor(source_seed_feature, dtype=torch.float).to(device)
target_seed_feature = torch.tensor(target_seed_feature, dtype=torch.float).to(device)
print(source_seed_feature.shape, target_seed_feature.shape)

# get torch tensor
seed_feature = torch.cat((source_seed_feature, target_seed_feature), dim=0).to(device)

# use k-means to cluster the keys into 10 clusters
kmeans = KMeans(50)
clusters = kmeans.fit_predict(seed_feature)

# show clusters
for i in range(50):
    print('cluster {}:'.format(i))
    print([source_keys_list[j] for j in range(len(source_keys_list)) if clusters[j] == i])
    print([target_keys_list[j] for j in range(len(target_keys_list)) if clusters[j + len(source_keys_list)] == i])
    print('\n')















# source_key_embeddings = model.encode(source_keys_list)
# target_key_embeddings = model.encode(target_keys_list)
# source_key_embeddings = torch.tensor(source_key_embeddings).to(device)
# target_key_embeddings = torch.tensor(target_key_embeddings).to(device)
# # get torch tensor
# all_key_embeddings = torch.cat((source_key_embeddings, target_key_embeddings), dim=0).to(device)

# # use k-means to cluster the keys into 10 clusters

# kmeans = KMeans(30)
# clusters = kmeans.fit_predict(all_key_embeddings)

# # show 10 clusters
# for i in range(30):
#     print('cluster {}:'.format(i))
#     print([source_keys_list[j] for j in range(len(source_keys_list)) if clusters[j] == i])
#     print([target_keys_list[j] for j in range(len(target_keys_list)) if clusters[j + len(source_keys_list)] == i])
#     print('\n')




