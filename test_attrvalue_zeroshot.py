from data.utils_yb import *
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

scores = scores_key * scores_value
with open('data/YAGO15K-FB15K/attrValue_scores_zeroshot_result2.txt', 'w') as f:
    for i in range(len(source_keyValue_sents)):
        f.write(source_keyValue_sents[i] + '\n')
        # print top 10 similar sentences with scores
        lis = [(target_keyValue_sents[j], scores[i][j]) for j in scores[i].argsort()[-10:][::-1]]
        # write to file
        for j in range(len(lis)):
            f.write(lis[j][0] + ' ' + str(lis[j][1]) + '|||')
        f.write('\n')
        f.write('\n')


# save scores, please avoid serializing a string larger than 4 GiB requires pickle
with open('data/YAGO15K-FB15K/attrValue_scores_zeroshot_result2.pkl', 'wb') as f:
    pickle.dump(scores, f)