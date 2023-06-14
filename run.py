import warnings
warnings.filterwarnings('ignore')

import os
import torch
import keras
import numpy as np
from utils import *
from tqdm import *
from evaluate import evaluate
from keras.layers import *
import tensorflow.compat.v1 as tf
import keras.backend as K
from layer import NR_GraphAttention
from os.path import join as pjoin
import sparse_eval
import pickle
from os.path import join as pjoin
from sinkhorn import matrix_sinkhorn





os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

seed = 12306
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

'''

----------------------- data information -------------------------
name             |   explain              |       type
___________________________________________________________________
dev_pair         | test pair              | [(1, 100), (2,103)]
adj_matrix       | (h,r,t) adj[h,t] = adj[t,h] = 1
r_index       有反向边 [[count, r], [count, r]] count = different (h, t) len(r_index) = num_triple
r_val           1/degree
adj_features    norm(adj+I)
rel_features    norm(rel_in || rel_out)
   

'''

attr_scores = np.load(pjoin('data', 'YAGO15K-FB15K', 'attr_scores_no_rule.pkl'), allow_pickle=True)
# norm attr scores
attr_scores = (attr_scores - np.min(attr_scores)) / (np.max(attr_scores) - np.min(attr_scores))

# image_scores = np.load(pjoin('data', 'DB15K-FB15K', 'image_scores.pkl'), allow_pickle=True)
# # norm image scores
# image_scores = (image_scores - np.min(image_scores)) / (np.max(image_scores) - np.min(image_scores))

use_image_cnt = 6
with open(pjoin('data', 'image_embed', 'image_score{}_yago15k_max_for_dev.npy'.format(use_image_cnt)), 'rb') as f:
    image_scores = pickle.load(f)

print('attr_scores.shape:', attr_scores.shape)
print('image_scores.shape:', image_scores.shape)



dataset = "YAGO15K-FB15K"
source = dataset.split('-')[0]
max_image_cnt = 1

if source == 'DB15K':
    source_image_embedding = np.load(pjoin('mmkb', 'DB15K', 'embedding', 'images_embeddings{}.npy'.format(max_image_cnt)), allow_pickle=True)
elif source == 'YAGO15K':
    source_image_embedding = np.load(pjoin('mmkb', 'YAGO15K', 'embedding', 'images_embeddings{}.npy'.format(max_image_cnt)), allow_pickle=True)

target_image_embedding = np.load(pjoin('mmkb', 'FB15K', 'embedding', 'images_embeddings{}.npy'.format(max_image_cnt)), allow_pickle=True)
image_embedding = np.concatenate([source_image_embedding, target_image_embedding], axis=0)
image_size, image_dim = image_embedding.shape
print('image_size:', image_size, 'image_dim:', image_dim)




train_pair, dev_pair, adj_matrix, r_index, r_val, adj_features, rel_features = load_data("data/" + dataset + '/', train_ratio=0.20)
adj_matrix = np.stack(adj_matrix.nonzero(), axis=1)
print(adj_matrix)
rel_matrix, rel_val = np.stack(rel_features.nonzero(), axis=1), rel_features.data
ent_matrix, ent_val = np.stack(adj_features.nonzero(), axis=1), adj_features.data
print(rel_matrix)
print(rel_val)
node_size = adj_features.shape[0]
rel_size = rel_features.shape[1]
triple_size = len(adj_matrix)
node_hidden = 128
rel_hidden = 128
batch_size = 1024
dropout_rate = 0.3
lr = 0.005
gamma = 1
depth = 2


def get_embedding(index_a, index_b, vec=None):
    if vec is None:
        inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix]
        # what the fuck
        inputs = [np.expand_dims(item, axis=0) for item in inputs]
        vec = get_emb.predict_on_batch(inputs)
    Lvec = np.array([vec[e] for e in index_a])
    Rvec = np.array([vec[e] for e in index_b])
    Lvec = Lvec / (np.linalg.norm(Lvec, axis=-1, keepdims=True) + 1e-5)
    Rvec = Rvec / (np.linalg.norm(Rvec, axis=-1, keepdims=True) + 1e-5)
    return Lvec, Rvec


class TokenEmbedding(keras.layers.Embedding):
    """Embedding layer with weights returned."""

    def compute_output_shape(self, input_shape):
        return self.input_dim, self.output_dim

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs):
        return self.embeddings


def get_trgat(node_hidden, rel_hidden, triple_size=triple_size, node_size=node_size, rel_size=rel_size, dropout_rate=0,
              gamma=3, lr=0.005, depth=2):
    adj_input = Input(shape=(None, 2))
    index_input = Input(shape=(None, 2), dtype='int64')
    val_input = Input(shape=(None,))
    rel_adj = Input(shape=(None, 2))
    ent_adj = Input(shape=(None, 2))
    # [adj_matrix, r_index, r_val, rel_matrix, ent_matrix]
    # Is that useful? val_input
    # non_image_node_size = node_size - image_size
    ent_emb = TokenEmbedding(node_size, node_hidden, trainable=True)(val_input)
    rel_emb = TokenEmbedding(rel_size, node_hidden, trainable=True)(val_input)
    # linear from image_dim to node_hidden, image_embedding is not trainable and linear is not trainable too
    # image_emb = Dense(node_hidden, use_bias=False, trainable=False)(image_embedding)
    # ent_emb = Concatenate(axis=0)([ent_emb, image_emb])


    def avg(tensor, size):
        adj = K.cast(K.squeeze(tensor[0], axis=0), dtype="int64")
        adj = tf.SparseTensor(indices=adj, values=tf.ones_like(adj[:, 0], dtype='float32'),
                              dense_shape=(node_size, size))
        adj = tf.sparse_softmax(adj)
        return tf.sparse_tensor_dense_matmul(adj, tensor[1])

    opt = [rel_emb, adj_input, index_input, val_input]
    ent_feature = Lambda(avg, arguments={'size': node_size})([ent_adj, ent_emb])
    rel_feature = Lambda(avg, arguments={'size': rel_size})([rel_adj, rel_emb])

    e_encoder = NR_GraphAttention(node_size, activation="tanh",
                                  rel_size=rel_size,
                                  use_bias=True,
                                  depth=depth,
                                  triple_size=triple_size)

    r_encoder = NR_GraphAttention(node_size, activation="tanh",
                                  rel_size=rel_size,
                                  use_bias=True,
                                  depth=depth,
                                  triple_size=triple_size)

    out_feature = Concatenate(-1)([e_encoder([ent_feature] + opt), r_encoder([rel_feature] + opt)])
    out_feature = Dropout(dropout_rate)(out_feature)

    alignment_input = Input(shape=(None, 2))

    def align_loss(tensor):
        def squared_dist(x):
            A, B = x
            row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
            row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.
            row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
            row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.
            return row_norms_A + row_norms_B - 2 * tf.matmul(A, B, transpose_b=True)

        emb = tensor[1]
        l, r = K.cast(tensor[0][0, :, 0], 'int32'), K.cast(tensor[0][0, :, 1], 'int32')
        l_emb, r_emb = K.gather(reference=emb, indices=l), K.gather(reference=emb, indices=r)

        pos_dis = K.sum(K.square(l_emb - r_emb), axis=-1, keepdims=True)
        r_neg_dis = squared_dist([r_emb, emb])
        l_neg_dis = squared_dist([l_emb, emb])

        l_loss = pos_dis - l_neg_dis + gamma
        l_loss = l_loss * (
                    1 - K.one_hot(indices=l, num_classes=node_size) - K.one_hot(indices=r, num_classes=node_size))

        r_loss = pos_dis - r_neg_dis + gamma
        r_loss = r_loss * (
                    1 - K.one_hot(indices=l, num_classes=node_size) - K.one_hot(indices=r, num_classes=node_size))

        r_loss = (r_loss - K.stop_gradient(K.mean(r_loss, axis=-1, keepdims=True))) / K.stop_gradient(
            K.std(r_loss, axis=-1, keepdims=True))
        l_loss = (l_loss - K.stop_gradient(K.mean(l_loss, axis=-1, keepdims=True))) / K.stop_gradient(
            K.std(l_loss, axis=-1, keepdims=True))

        lamb, tau = 30, 10
        # l_loss = K.log(K.sum(K.exp(lamb * l_loss + tau), axis=-1))
        # r_loss = K.log(K.sum(K.exp(lamb * r_loss + tau), axis=-1))
        # l_loss = K.logsumexp(lamb * l_loss + tau, axis=-1)
        # r_loss = K.logsumexp(lamb * r_loss + tau, axis=-1)
        l_loss = tf.reduce_logsumexp(lamb * l_loss + tau, axis=-1)
        r_loss = tf.reduce_logsumexp(lamb * r_loss + tau, axis=-1)
        final_loss = K.mean(l_loss + r_loss)

        print(final_loss)
        return final_loss

    loss = Lambda(align_loss)([alignment_input, out_feature])

    inputs = [adj_input, index_input, val_input, rel_adj, ent_adj]
    train_model = keras.Model(inputs=inputs + [alignment_input], outputs=loss)
    train_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=keras.optimizers.RMSprop(lr))

    feature_model = keras.Model(inputs=inputs, outputs=out_feature)

    return train_model, feature_model

model,get_emb = get_trgat(dropout_rate=dropout_rate,
                          node_size=node_size,
                          rel_size=rel_size,
                          depth=depth,
                          gamma=gamma,
                          node_hidden=node_hidden,
                          rel_hidden=rel_hidden,
                          lr=lr)

evaluater = evaluate(dev_pair)
model.summary()

rest_set_1 = [e1 for e1, e2 in dev_pair]
rest_set_2 = [e2 for e1, e2 in dev_pair]
sourceId2Index = {e1: i for i, e1 in enumerate(rest_set_1)}
targetId2Index = {e2: i for i, e2 in enumerate(rest_set_2)}
np.random.shuffle(rest_set_1)
np.random.shuffle(rest_set_2)

epoch = 20
for turn in range(5):
    for i in trange(epoch):
        np.random.shuffle(train_pair)
        for pairs in [train_pair[i * batch_size:(i + 1) * batch_size] for i in
                      range(len(train_pair) // batch_size + 1)]:
            if len(pairs) == 0:
                continue
            inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix, pairs]
            # what the fuck
            inputs = [np.expand_dims(item, axis=0) for item in inputs]
            output = model.train_on_batch(inputs, np.zeros((1, 1)))
            print(output)
        if i%5==4 :
            Lvec, Rvec = get_embedding(dev_pair[:, 0], dev_pair[:, 1])
            # get score matrix for dev_pair
            scores = torch.Tensor(Lvec.dot(Rvec.T) + image_scores)
            scores = matrix_sinkhorn(1 - scores)
            sparse_eval.evaluate_sim_matrix(link = torch.stack([torch.arange(len(dev_pair)), 
                                                    torch.arange(len(dev_pair))], dim=0),
                                                    sim_x2y=scores,
                                                    no_csls=True)
            # save score
            # pickle.dump(scores, open(pjoin('data', 'DB15K-FB15K', 'gnn_scores.pkl'), 'wb'))
            # evaluater.test(Lvec, Rvec)
        new_pair = []
    Lvec, Rvec = get_embedding(rest_set_1, rest_set_2)
    attr_scores_now = np.zeros((len(rest_set_1), len(rest_set_2)), dtype=np.float32)
    image_scores_now = np.zeros((len(rest_set_1), len(rest_set_2)), dtype=np.float32)
    for i, e1 in enumerate(rest_set_1):
        for j, e2 in enumerate(rest_set_2):
            source_index = sourceId2Index[e1]
            target_index = targetId2Index[e2]
            attr_scores_now[i, j] = attr_scores[source_index, target_index]
            image_scores_now[i, j] = image_scores[source_index, target_index]
    
    # image_scores_now = np.zeros((len(rest_set_1), len(rest_set_2)), dtype=np.float32)
    # for i, e1 in enumerate(rest_set_1):
    #     for j, e2 in enumerate(rest_set_2):
    #         source_index = sourceId2Index[e1]
    #         target_index = targetId2Index[e2]
    #         image_scores_now[i, j] = image_scores[source_index, target_index]

    # A, B = evaluater.CSLS_cal(Lvec, Rvec, False)
    scores = Lvec.dot(Rvec.T) + image_scores_now
    A, B = scores.argmax(axis=0), scores.argmax(axis=1)
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