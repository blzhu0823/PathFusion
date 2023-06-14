from data.utils_db import *
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import pickle

source_keyValue_sents = sorted(list(source_attr_value_set))
target_keyValue_sents = sorted(list(target_attr_value_set))









device = torch.device('cuda:5') if torch.cuda.is_available() else torch.device('cpu')
model = SentenceTransformer('Maite89/Roberta_finetuning_semantic_similarity_stsb_multi_mt').to(device)




source_key_embeddings = []
target_key_embeddings = []
source_value = []
target_value = []

batch_size = 128
for i in range(0, len(source_keyValue_sents), batch_size):
    key_sents = source_keyValue_sents[i:i + batch_size]
    source_key_embeddings.append(model.encode(key_sents))
source_key_embeddings = np.concatenate(source_key_embeddings, axis=0)

for i in tqdm(range(0, len(target_keyValue_sents), batch_size)):
    key_sents = target_keyValue_sents[i:i + batch_size]
    target_key_embeddings.append(model.encode(key_sents))
target_key_embeddings = np.concatenate(target_key_embeddings, axis=0)

scores_key = np.matmul(source_key_embeddings, target_key_embeddings.T)

scores = scores_key


# save scores, please avoid serializing a string larger than 4 GiB requires pickle
with open('data/DB15K-FB15K/tmp.pkl', 'wb') as f:
    pickle.dump(scores, f)