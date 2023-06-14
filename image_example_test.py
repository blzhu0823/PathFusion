from transformers import AutoImageProcessor, SwinModel
import torch
import os
import numpy as np
import time
from PIL import Image


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


batch_size = 16


image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-large-patch4-window12-384-in22k")
model = SwinModel.from_pretrained("microsoft/swin-large-patch4-window12-384-in22k").to('cuda')


def open_image_and_encode(image_path):
    image = Image.open(image_path)
    image.thumbnail((500, 500), Image.ANTIALIAS)
    image = image.convert('RGB')
    inputs = image_processor(image, return_tensors="pt").to('cuda')
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.pooler_output
    return embeddings



source_images = []
target_images = []

for i in range(1, 7):
    source_images.append(open_image_and_encode('./例子1/db15k/{}.jpg'.format(i)))
    target_images.append(open_image_and_encode('./例子1/fb15k/{}.jpg'.format(i)))


# calculate the similarity

similarity = torch.zeros((6, 6))

for i in range(6):
    for j in range(6):
        similarity[i][j] = torch.cosine_similarity(source_images[i], target_images[j], dim=1)


# row level softmax

similarity = torch.softmax(similarity, dim=1)

print(similarity)



