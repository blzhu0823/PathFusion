import numpy as np
import scipy.sparse as sp
import scipy
import os
import multiprocessing
from collections import defaultdict

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).T

def load_triples(file_name):
    triples = []
    entity = set()
    rel = set([0])
    for line in open(file_name, 'r'):
        head, r, tail = [int(item) for item in line.split()]
        entity.add(head); entity.add(tail); rel.add(r+1)
        triples.append((head, r+1, tail))
    return entity, rel, triples

def load_triples_tkg(file_name):
    triples = []
    entity = set()
    rel = set([0])
    time = set([0])
    entity2time = defaultdict(int)
    

    for line in open(file_name, 'r'):
        para = line.split()
        if len(para) == 5:
            head, r, tail, ts, te = [int(item) for item in para]
            entity.add(head);
            entity.add(tail);
            rel.add(r + 1)
            time.add(ts + 1);
            time.add(te + 1)
            triples.append((head, r + 1, tail))
            entity2time[(head, ts + 1)] += 1
            entity2time[(tail, ts + 1)] += 1
            entity2time[(head, te + 1)] += 1
            entity2time[(tail, te + 1)] += 1
        else:
            head, r, tail, t = [int(item) for item in para]
            entity.add(head);
            entity.add(tail);
            rel.add(r + 1)
            time.add(t + 1)
            triples.append((head, r + 1, tail))
            entity2time[(head, t + 1)] += 1
            entity2time[(tail, t + 1)] += 1


    return entity, rel, triples, time, entity2time

def load_alignment_pair(file_name):
    alignment_pair = []
    c = 0
    for line in open(file_name,'r'):
        e1,e2 = line.split()
        alignment_pair.append((int(e1), int(e2)))
    return alignment_pair

def get_matrix(triples,entity,rel):
        ent_size = max(entity) + 1
        rel_size = (max(rel) + 1)
        print(ent_size, rel_size)
        adj_matrix = sp.lil_matrix((ent_size, ent_size))
        adj_features = sp.lil_matrix((ent_size, ent_size))
        radj = []
        rel_in = np.zeros((ent_size, rel_size))
        rel_out = np.zeros((ent_size, rel_size))
        
        for i in range(max(entity)+1):
            adj_features[i, i] = 1

        for h, r, t in triples:
            adj_matrix[h, t] = 1; adj_matrix[t, h] = 1
            adj_features[h, t] = 1; adj_features[t, h] = 1
            radj.append([h, t, r]); radj.append([t, h, r+rel_size])
            rel_out[h][r] += 1; rel_in[t][r] += 1
            
        count = -1
        s = set()
        d = {}
        r_index, r_val = [], []
        for h, t, r in sorted(radj, key=lambda x: x[0]*10e10+x[1]*10e5):
            if ' '.join([str(h), str(t)]) in s:
                r_index.append([count, r])
                r_val.append(1)
                d[count] += 1
            else:
                count += 1
                d[count] = 1
                s.add(' '.join([str(h), str(t)]))
                r_index.append([count, r])
                r_val.append(1)
        for i in range(len(r_index)):
            r_val[i] /= d[r_index[i][0]]
        
        rel_features = np.concatenate([rel_in, rel_out], axis=1)
        adj_features = normalize_adj(adj_features)
        rel_features = normalize_adj(sp.lil_matrix(rel_features))    
        return adj_matrix, r_index, r_val, adj_features,rel_features
    
def load_data(lang, train_ratio = 0.3):
    entity1, rel1, triples1 = load_triples(lang + 'triples_1')
    entity2, rel2, triples2 = load_triples(lang + 'triples_2')
    # modified here #
    # if "_en" in lang:
    if 'FB15K' in lang:
        alignment_pair = load_alignment_pair(lang + 'ref_ent_ids')
        np.random.shuffle(alignment_pair)
        # train_pair, dev_pair = alignment_pair[0:int(len(alignment_pair)*train_ratio)], alignment_pair[int(len(alignment_pair)*train_ratio):]
        train_pair = load_alignment_pair(lang + 'sup_ent_ids')
        dev_pair = load_alignment_pair(lang + 'dev_ent_ids')
    else:
        train_pair = load_alignment_pair(lang + 'sup_ent_ids')
        dev_pair = load_alignment_pair(lang + 'ref_ent_ids')
        ae_features = None
    
    adj_matrix, r_index, r_val, adj_features, rel_features = get_matrix(triples1+triples2,entity1.union(entity2),rel1.union(rel2))

    return np.array(train_pair),np.array(dev_pair), adj_matrix, np.array(r_index),np.array(r_val),adj_features,rel_features


def load_data_tkg(lang, train_ratio=0.3):
    entity1, rel1, triples1, time1, entity2time1 = load_triples_tkg(lang + 'triples_1')
    entity2, rel2, triples2, time2, entity2time2 = load_triples_tkg(lang + 'triples_2')
    # modified here #

    train_pair = load_alignment_pair(lang + 'sup_pairs')
    dev_pair = load_alignment_pair(lang + 'ref_pairs')
    if train_ratio < 0.25:
        train_ratio = int(len(train_pair) * train_ratio)
        dev_pair = train_pair[train_ratio:] + dev_pair
        train_pair = train_pair[:train_ratio]
        print(len(train_pair))
    
    
    # get time features for dev pairs
    max_time = max(time1.union(time2))
    min_time = min(time1.union(time2))
    source2time = np.zeros((len(dev_pair), max_time + 1))
    target2time = np.zeros((len(dev_pair), max_time + 1))
    for i, ent1 in enumerate(np.array(dev_pair)[:, 0]):
        for t in range(max_time + 1):
            source2time[i, t] = entity2time1[(ent1, t)]
    for i, ent2 in enumerate(np.array(dev_pair)[:, 1]):
        for t in range(max_time + 1):
            target2time[i, t] = entity2time2[(ent2, t)]
    
    time_features = source2time.dot(target2time.T)
    # normalize time features
    # time_features = (time_features - np.min(time_features)) / (np.max(time_features) - np.min(time_features))
    # row max min normalization
    for i in range(len(time_features)):
        time_features[i] = (time_features[i] - np.min(time_features[i])) / (np.max(time_features[i]) - np.min(time_features[i]))
    source2devindex = {ent1: i for i, ent1 in enumerate(np.array(dev_pair)[:, 0])}
    target2devindex = {ent2: i for i, ent2 in enumerate(np.array(dev_pair)[:, 1])}

    adj_matrix, r_index, r_val, adj_features, rel_features = get_matrix(triples1+triples2,entity1.union(entity2),rel1.union(rel2))

    return np.array(train_pair),np.array(dev_pair), adj_matrix, np.array(r_index),np.array(r_val),adj_features,rel_features, time_features, source2devindex, target2devindex
