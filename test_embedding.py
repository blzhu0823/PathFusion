import pickle
import numpy as np
from tqdm import tqdm
import os




source = 'yago15k'
target = 'fb15k'
use_img_num = 6

print('fuck:', use_img_num)


if source == 'db15k':
    from data.utils_db import *
elif source == 'yago15k':
    from data.utils_yb import *



print()
print()
# load source image embedding
with open('./data/image_embed/{}.npy'.format(source), 'rb') as f:
    source_embedding = pickle.load(f)

# load source entity id to image id mapping
with open('./data/image_embed/{}'.format(source), 'rb') as f:
    source_id2img = pickle.load(f)


print('source embedding shape: {}'.format(source_embedding.shape))
print('source id2img length: {}'.format(len(source_id2img)))


# load target image embedding
with open('./data/image_embed/{}.npy'.format(target), 'rb') as f:
    target_embedding = pickle.load(f)

# load target entity id to image id mapping
with open('./data/image_embed/{}'.format(target), 'rb') as f:
    target_id2img = pickle.load(f)


print('target embedding shape: {}'.format(target_embedding.shape))
print('target id2img length: {}'.format(len(target_id2img)))




print('source target entity num:', len(source2id), len(target2id))
# score init as -float('inf')
score = np.zeros((len(source2id), len(target2id)))

for i in tqdm(range(len(source2id))):
    for j in range(len(target2id)):
        for ii in range(min(use_img_num, len(source_id2img[i]))):
            for jj in range(min(use_img_num, len(target_id2img[j]))):
                # score[i, j] = max(score[i, j], np.dot(source_embedding[source_id2img[i][ii]], target_embedding[target_id2img[j][jj]]))
                score[i, j] += np.dot(source_embedding[source_id2img[i][ii]], target_embedding[target_id2img[j][jj]])




# save score to data/image_embed/image_score{use_img_num}.npy
score.dump('./data/image_embed/image_score{}_{}_sum.npy'.format(use_img_num, source))

