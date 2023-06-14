from transformers import BertTokenizer, BertModel
from data.utils_db import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch
import random
from text_matching.inference_pointwise import test_inference
from sparse_eval import evaluate_sim_matrix
from tqdm import tqdm


def softmax(x, dim = -1):
    x = x - torch.max(x, dim=dim, keepdim=True)[0]
    x = torch.exp(x)
    x = x / torch.sum(x, dim=dim, keepdim=True)
    return x

# device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
# model = SentenceTransformer('Maite89/Roberta_finetuning_semantic_similarity_stsb_multi_mt')
# print(model.encode(['This is a test sentence', 'This is a test sentence']).shape)


# def get_embedding(index_a, index_b, vec):
#     Lvec = np.array([vec[e] for e in index_a])
#     Rvec = np.array([vec[min(e + 1, len(vec) - 1)] for e in index_b])
#     # normalize
#     Lvec = Lvec / (np.linalg.norm(Lvec, axis=-1, keepdims=True) + 1e-5)
#     Rvec = Rvec / (np.linalg.norm(Rvec, axis=-1, keepdims=True) + 1e-5)

#     return Lvec, Rvec


# # using bert for set matching
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
# # add a head to cls for set matching
# model.add_module('cls_head', nn.Linear(768, 1))

# # encode the sentences
# def encode_sentence(sent1, sent2):
#     encoded_dict = tokenizer.encode_plus(
#         text=sent1,
#         text_pair=sent2,
#         add_special_tokens=True,
#         max_length=128,
#         pad_to_max_length=True,
#         return_attention_mask=True,
#         return_tensors='pt'
#     )
#     return encoded_dict

def source_prompt(attr_value_schemas):
    if len(attr_value_schemas) == 0:
        return 'No attributes'
    prompt = ''
    for attr_value_schema in attr_value_schemas[:-1]:
        attr, value, schema = attr_value_schema
        prompt += '{}: {} {}, '.format(attr, value, schema)
    
    attr, value, schema = attr_value_schemas[-1]
    prompt += '{}: {} {}'.format(attr, value, schema)
    return prompt

def target_prompt(attr_values):
    if len(attr_values) == 0:
        return 'No attributes'
    prompt = ''
    for attr_value in attr_values[:-1]:
        attr, value = attr_value
        prompt += '{}: {}, '.format(attr, value)
    
    attr, value = attr_values[-1]
    prompt += '{}: {}'.format(attr, value)
    return prompt


# seed_num = len(seeds)
# train_pair, dev_pair = [], []
# train_ratio = 0.2
# train_num = int(seed_num * train_ratio)
# for i in range(seed_num):
#     seed = seeds[i]
#     source_ent, target_ent = seed[0], seed[1]
#     source_attrs = id2attrs[source_ent]
#     target_attrs = id2attrs[target_ent]
#     source_sent = source_prompt(source_attrs)
#     target_sent = target_prompt(target_attrs)
#     if source_sent == 'No attributes' or target_sent == 'No attributes' or len(train_pair) >= train_num:
#         dev_pair.append((source_ent, target_ent))
#         continue
#     train_pair.append((source_ent, target_ent))

# print('train pair num: {}, dev pair num: {}'.format(len(train_pair), len(dev_pair)))


# #-----------------do pointwise sim evaluation----------------
from rich import print

import torch
from transformers import AutoTokenizer

device = 'cuda:0'
tokenizer = AutoTokenizer.from_pretrained('./text_matching/checkpoints/attr/model_best/')
model = torch.load('./text_matching/checkpoints/attr/model_best/model.pt')
model.to(device).eval()
def test_inference(text1, text2, max_seq_len=128) -> torch.tensor:
    """
    预测函数，输入两句文本，返回这两个文本相似/不相似的概率。

    Args:
        text1 (str): 第一段文本
        text2 (_type_): 第二段文本
        max_seq_len (int, optional): 文本最大长度. Defaults to 128.
    
    Reuturns:
        torch.tensor: 相似/不相似的概率 -> (batch, 2)
    """
    encoded_inputs = tokenizer(
        text=[*text1],
        text_pair=[*text2],
        truncation=True,
        max_length=max_seq_len,
        return_tensors='pt',
        padding='max_length')
    
    with torch.no_grad():
        model.eval()
        logits = model(input_ids=encoded_inputs['input_ids'].to(device),
                        token_type_ids=encoded_inputs['token_type_ids'].to(device),
                        attention_mask=encoded_inputs['attention_mask'].to(device))
    
    return softmax(logits)[:, 1]



dev_pair = np.array(dev_pair)
train_pair = np.array(train_pair)
dev_pair = dev_pair[-320:]


source_id2evalid = {}
source_evalid2id = {}
target_id2evalid = {}
target_evalid2id = {}
for i, (source_id, target_id) in enumerate(dev_pair):
    source_id2evalid[source_id] = i
    source_evalid2id[i] = source_id
    target_id2evalid[target_id] = i
    target_evalid2id[i] = target_id


batch_size = 256
similarity = []
for source_id in tqdm(dev_pair[:, 0]):
    source_sent = source_prompt(id2attrs[int(source_id)])
    sim_row = []
    # a batch: batch_size
    for i in range(0, len(dev_pair), batch_size):
        targit_ids = dev_pair[i: i + batch_size, 1]
        target_sents = [target_prompt(id2attrs[int(target_id)]) for target_id in targit_ids]
        logits = test_inference([source_sent] * len(target_sents), target_sents)
        sim_row.extend(logits.cpu().numpy())
    similarity.append(sim_row)

similarity = np.array(similarity)
print(similarity.shape)
print(similarity)


evaluate_sim_matrix(torch.stack([torch.arange(len(similarity)), torch.arange(len(similarity))]).to(torch.long), sim_x2y=torch.Tensor(similarity), sim_y2x=torch.Tensor(similarity.T), no_csls=False)

# save similarity matrix

# similarity.dump('./data/DB15K-FB15K/similarity_attr.npy')















    




#-----------------implement matching by myself(dataset and dataloader)----------------
# # establish dataset
# class myDataset(Dataset):
#     def __init__(self, data):
#         self.data = data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         source_ent, target_ent = self.data[index]
#         source_attrs = id2attrs[source_ent]
#         target_attrs = id2attrs[target_ent]
#         source_sent = source_prompt(source_attrs)
#         target_sent = target_prompt(target_attrs)
#         return encode_sentence(source_sent, target_sent)


# train_dataset = myDataset(train_pair)
# dev_dataset = myDataset(dev_pair)

# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# dev_dataloader = DataLoader(dev_dataset, batch_size=32, shuffle=True)


# # train


# def train():
#     model.train()
#     for batch in train_dataloader:
#         input_ids = batch['input_ids']
#         attention_mask = batch['attention_mask']
#         outputs = model(input_ids.squeeze(1), attention_mask=attention_mask)
#         # use cls embedding
#         cls_embedding = outputs[0][:, 0, :]




# --------------generate train dev set for attr-value matching----------------

# training_set = []




# for i in range(len(train_pair)):
#     source_ent, target_ent = train_pair[i]
#     source_attrs = id2attrs[source_ent]
#     target_attrs = id2attrs[target_ent]
#     source_sent = source_prompt(source_attrs)
#     target_sent = target_prompt(target_attrs)
#     training_set.append((source_sent, target_sent, 1))

# for i in range(len(train_pair)):
#     # do negative sampling
#     i = random.randint(0, len(source2id) - 1)
#     j = random.randint(len(source2id), len(target2id) - 1)
#     source_attrs = id2attrs[i]
#     target_attrs = id2attrs[j]
#     source_sent = source_prompt(source_attrs)
#     target_sent = target_prompt(target_attrs)
#     training_set.append((source_sent, target_sent, 0))

# random.shuffle(training_set)
# # save training set
# with open('text_matching/data/attr/db15k/train.txt', 'w') as f:
#     for i in range(len(training_set)):
#         source_sent, target_sent, label = training_set[i]
#         f.write('{}\t{}\t{}\n'.format(source_sent, target_sent, label))

# dev_set = []
# for i in range(len(dev_pair)):
#     source_ent, target_ent = dev_pair[i]
#     source_attrs = id2attrs[source_ent]
#     target_attrs = id2attrs[target_ent]
#     source_sent = source_prompt(source_attrs)
#     target_sent = target_prompt(target_attrs)
#     dev_set.append((source_sent, target_sent, 1))

# for i in range(len(dev_pair)):
#     # do negative sampling
#     i = random.randint(0, len(source2id) - 1)
#     j = random.randint(len(source2id), len(target2id) - 1)
#     source_attrs = id2attrs[i]
#     target_attrs = id2attrs[j]
#     source_sent = source_prompt(source_attrs)
#     target_sent = target_prompt(target_attrs)
#     dev_set.append((source_sent, target_sent, 0))

# random.shuffle(dev_set)
# # save dev set
# with open('text_matching/data/attr/db15k/dev.txt', 'w') as f:
#     for i in range(len(dev_set)):
#         source_sent, target_sent, label = dev_set[i]
#         f.write('{}\t{}\t{}\n'.format(source_sent, target_sent, label))

# # save train_pair and dev_pair
# with open('data/DB15K-FB15K/sup_ent_ids', 'w') as f:
#     for ent1, ent2 in train_pair:
#         f.write('{}\t{}\n'.format(ent1, ent2))

# with open('data/DB15K-FB15K/dev_ent_ids', 'w') as f:
#     for ent1, ent2 in dev_pair:
#         f.write('{}\t{}\n'.format(ent1, ent2))
    




#-----------------zeroshot for attr-value matching----------------

# sents = []
# for i in range(len(id2attrs)):
#     if i < len(source2id):
#         sents.append(source_prompt(id2attrs[i]))
#     else:
#         sents.append(target_prompt(id2attrs[i]))

# embeddings = []
# batch_size = 32
# for i in range(0, len(sents), batch_size):
#     embeddings.append(model.encode(sents[i:i + batch_size]))
# embeddings = np.concatenate(embeddings, axis=0)


# Lvec, Rvec = get_embedding(dev_pair[:, 0], dev_pair[:, 1], embeddings)

# evaluater.test(Lvec, Rvec)
