# from mmkb.DB15K.image_processing import mydataset as db15k_image_dataset
# from mmkb.YAGO15K.image_processing import mydataset as yago15k_image_dataset
from mmkb.FB15K.image_processing import mydataset as fb15k_image_dataset
from transformers import AutoImageProcessor, SwinModel
import torch
from datasets import load_dataset
import os
import numpy as np
import time


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


batch_size = 16


image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-large-patch4-window12-384-in22k")
model = SwinModel.from_pretrained("microsoft/swin-large-patch4-window12-384-in22k").to('cuda')
# images_embeddings = []

# time_start = time.time()
# for i in range(0, len(yago15k_image_dataset), batch_size):
#     batch = yago15k_image_dataset[i:i+batch_size]
#     ent_ids = batch[0]
#     images = batch[1]
#     ent_names = batch[2]
#     inputs = image_processor(images, return_tensors="pt").to('cuda')
#     model.eval()
#     with torch.no_grad():
#         outputs = model(**inputs)
#         embeddings = outputs.pooler_output
#         images_embeddings.append(embeddings)
# time_end = time.time()
# print('time cost', time_end - time_start, 's')
    
# images_embeddings = torch.cat(images_embeddings, dim=0).cpu().numpy()
# print(images_embeddings.shape)
# # save the embeddings (numpy format)
# images_embeddings.dump('./mmkb/YAGO15K/embedding/images_embeddings1.npy')



# time_start = time.time()
# images_embeddings = []
# for i in range(0, len(db15k_image_dataset), batch_size):
#     batch = db15k_image_dataset[i:i+batch_size]
#     ent_ids = batch[0]
#     images = batch[1]
#     ent_names = batch[2]
#     inputs = image_processor(images, return_tensors="pt").to('cuda')
#     model.eval()
#     with torch.no_grad():
#         outputs = model(**inputs)
#         embeddings = outputs.pooler_output
#         images_embeddings.append(embeddings)
# time_end = time.time()
# print('time cost', time_end - time_start, 's')
    
# images_embeddings = torch.cat(images_embeddings, dim=0).cpu().numpy()
# print(images_embeddings.shape)
# # save the embeddings (numpy format)
# images_embeddings.dump('./mmkb/DB15K/embedding/images_embeddings1.npy')



time_start = time.time()
images_embeddings = []
for i in range(0, len(fb15k_image_dataset), batch_size):
    batch = fb15k_image_dataset[i:i+batch_size]
    ent_ids = batch[0]
    images = batch[1]
    ent_names = batch[2]
    inputs = image_processor(images, return_tensors="pt").to('cuda')
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.pooler_output
        images_embeddings.append(embeddings)
time_end = time.time()
print('time cost', time_end - time_start, 's')
    
images_embeddings = torch.cat(images_embeddings, dim=0).cpu().numpy()
print(images_embeddings.shape)
# save the embeddings (numpy format)
images_embeddings.dump('./mmkb/FB15K/embedding/images_embeddings1.npy')