from dgl_version import *
import dgl
import numpy as np
import torch
import time
from utils import *
from evaluate import evaluate
from tqdm import *
# import tensorflow as tf


# gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
# config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
# # config.gpu_options.per_process_gpu_memory_fraction = 0.3
# sess = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(sess)

train_pair, dev_pair, adj_matrix, r_index, r_val, adj_features, rel_features = load_data("data/DB15K-FB15K/", train_ratio=0.20)
adj_matrix = np.stack(adj_matrix.nonzero(), axis=1)
rel_matrix, rel_val = np.stack(rel_features.nonzero(), axis=1), rel_features.data
ent_matrix, ent_val = np.stack(adj_features.nonzero(), axis=1), adj_features.data

node_size = adj_features.shape[0]
rel_size = rel_features.shape[1]
triple_size = len(adj_matrix) # not triple size, but number of diff(h, t)
node_hidden = 128
rel_hidden = 128
batch_size = 1024
dropout_rate = 0.3
lr = 0.005
gamma = 1
depth = 2
device = 'cuda:7'

print('new_version')
def get_embedding(index_a, index_b, vec):
    vec = vec.detach().numpy()
    Lvec = np.array([vec[e] for e in index_a])
    Rvec = np.array([vec[e] for e in index_b])
    Lvec = Lvec / (np.linalg.norm(Lvec, axis=-1, keepdims=True) + 1e-5)
    Rvec = Rvec / (np.linalg.norm(Rvec, axis=-1, keepdims=True) + 1e-5)
    return Lvec, Rvec

def align_loss(align_input, embedding):
    def squared_dist(x):
        A, B = x
        row_norms_A = torch.sum(torch.square(A), dim=1)
        row_norms_A = torch.reshape(row_norms_A, [-1, 1])  # Column vector.
        row_norms_B = torch.sum(torch.square(B), dim=1)
        row_norms_B = torch.reshape(row_norms_B, [1, -1])  # Row vector.
        return row_norms_A + row_norms_B - 2 * torch.matmul(A, torch.transpose(B, 0, 1))
    # modified
    left = torch.tensor(align_input[:, 0])
    right = torch.tensor(align_input[:, 1])
    l_emb = embedding[left]
    r_emb = embedding[right]
    pos_dis = torch.sum(torch.square(l_emb - r_emb), dim=-1, keepdim=True)
    r_neg_dis = squared_dist([r_emb, embedding])
    l_neg_dis = squared_dist([l_emb, embedding])

    l_loss = pos_dis - l_neg_dis + gamma
    l_loss = l_loss * (1 - F.one_hot(left, num_classes=node_size) - F.one_hot(right, num_classes=node_size)).to(device)

    r_loss = pos_dis - r_neg_dis + gamma
    r_loss = r_loss * (1 - F.one_hot(left, num_classes=node_size) - F.one_hot(right, num_classes=node_size)).to(device)
    # modified
    with torch.no_grad():
        r_mean = torch.mean(r_loss, dim=-1, keepdim=True)
        r_std= torch.std(r_loss, dim=-1, keepdim=True)
        r_loss.data =(r_loss.data -r_mean)/r_std
        l_mean = torch.mean(l_loss, dim=-1, keepdim=True)
        l_std= torch.std(l_loss, dim=-1, keepdim=True)
        l_loss.data =(l_loss.data -l_mean)/l_std
        # l_loss = (l_loss - torch.mean(l_loss, dim=-1, keepdim=True).detach()) / torch.std(
        #     l_loss, dim=-1, keepdim=True).detach()

    lamb, tau = 30, 10
    l_loss = torch.logsumexp(lamb * l_loss + tau, dim=-1)
    r_loss = torch.logsumexp(lamb * r_loss + tau, dim=-1)
    return torch.mean(l_loss + r_loss)

def constructRelGraph(r_index):
    src, trg = [], []
    for index in r_index:
        src.append(index[1])
        trg.append(index[0])
    return dgl.heterograph({
        ('relation', 'in', 'index'): (torch.tensor(src), torch.tensor(trg)), })


def constructGraph(adj_matrix):
    src, trg = adj_matrix[:, 0], adj_matrix[:, 1]
    # todo src trg
    g = dgl.graph((src, trg))
    # g = dgl.graph(( trg,src))
    return g


def constructNode_Rel_interact(rel_matrix):
    src, trg = [], []
    for index in rel_matrix:
        src.append(index[1])
        trg.append(index[0])
    return dgl.heterograph({
        ('relation', 'link', 'entity'): (torch.tensor(src), torch.tensor(trg)), })

g = constructGraph(adj_matrix)
g_r = constructRelGraph(r_index)
n_r = constructNode_Rel_interact(rel_matrix)

print('begin')

model = overAll(node_size=node_size, node_hidden=node_hidden,
                 rel_size=rel_size, dropout_rate=dropout_rate,
                depth=depth, device=device)
model = model.to(device)
# opt = torch.optim.RMSprop(model.parameters(), lr=lr)
opt = torch.optim.Adam(model.parameters(), lr=lr)
print('model constructed')

evaluater = evaluate(dev_pair)
rest_set_1 = [e1 for e1, e2 in dev_pair]
rest_set_2 = [e2 for e1, e2 in dev_pair]
np.random.shuffle(rest_set_1)
np.random.shuffle(rest_set_2)


# here is dual-m
epoch = 20
for turn in range(5):
    for i in trange(epoch):
        np.random.shuffle(train_pair)
        for pairs in [train_pair[i * batch_size:(i + 1) * batch_size] for i in
                      range(len(train_pair) // batch_size + 1)]:
            output = model(g, g_r, n_r)
            # print(output)
            loss = align_loss(pairs, output)
            print(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if i %5 == 4:
            model.eval()
            output = model(g, g_r, n_r)
            Lvec, Rvec = get_embedding(dev_pair[:, 0], dev_pair[:, 1], output.cpu())
            evaluater.test(Lvec, Rvec)
            model.train()
        new_pair = []
    Lvec, Rvec = get_embedding(rest_set_1, rest_set_2, output.cpu())
    A, B = evaluater.CSLS_cal(Lvec, Rvec, False)
    for i, j in enumerate(A):
        if B[j] == i:
            new_pair.append([rest_set_1[j], rest_set_2[i]])

    train_pair = np.concatenate([train_pair, np.array(new_pair)], axis=0)
    for e1, e2 in new_pair:
        if e1 in rest_set_1:
            rest_set_1.remove(e1)

    for e1, e2 in new_pair:
        if e2 in rest_set_2:
            rest_set_2.remove(e2)
    epoch = 5