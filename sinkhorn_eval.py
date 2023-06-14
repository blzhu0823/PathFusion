import numba as nb
import numpy as np
from tqdm import tqdm
from scipy import optimize
import tensorflow as tf
import json
import os

seed = 12345
np.random.seed(seed)

# Sinkhorn operation
def sinkhorn(sims, eps=1e-6):
    sims = tf.exp(sims * 50)
    for k in range(10):
        sims = sims / (tf.reduce_sum(sims, axis=1, keepdims=True) + eps)
        sims = sims / (tf.reduce_sum(sims, axis=0, keepdims=True) + eps)
    return sims


def test(sims,mode = "sinkhorn", batch_size = 1024):
    if mode == "sinkhorn":
        results = []
        for epoch in range(len(sims) // batch_size + 1):
            sim = sims[epoch*batch_size:(epoch+1)*batch_size]
            rank = tf.argsort(-sim, axis=-1)
            ans_rank = np.array([i for i in range(epoch * batch_size, min((epoch+1) * batch_size, len(sims)))])
            x = np.expand_dims(ans_rank, axis=1)
            y = tf.tile(x, [1, len(sims)])
            results.append(tf.where(tf.equal(tf.cast(rank, ans_rank.dtype), tf.tile(np.expand_dims(ans_rank, axis=1), [1, len(sims)]))).numpy())
        results = np.concatenate(results, axis=0)
        
        @nb.jit(nopython=True)
        def cal(results):
            hits1, hits10, mrr = 0, 0, 0
            for x in results[:, 1]:
                if x < 1:
                    hits1 += 1
                if x < 10:
                    hits10 += 1
                mrr += 1/(x + 1)
            return hits1, hits10, mrr
        hits1, hits10, mrr = cal(results)
        print("hits@1 : %.2f%% hits@10 : %.2f%% MRR : %.2f%%" % (hits1/len(sims)*100, hits10/len(sims)*100, mrr/len(sims)*100))
        return hits1/len(sims), hits10/len(sims), mrr/len(sims)
    else:
        c = 0
        for i, j in enumerate(sims[1]):
            if i == j:
                c += 1
        print("hits@1 : %.2f%%" %(100 * c/len(sims[0])))
        return c/len(sims[0])

