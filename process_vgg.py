import h5py
import numpy as np


db15k_embedding = h5py.File('./image_vgg/DB15K_ImageData.h5', 'r')
yago15k_embedding = h5py.File('./image_vgg/YAGO15K_ImageData.h5', 'r')
fb15k_embedding = h5py.File('./image_vgg/FB15K_ImageData.h5', 'r')

import os
from os.path import join as pjoin



source_dbs = ['DB15K', 'YAGO15K']
source_image_embeddings = {'DB15K': db15k_embedding, 'YAGO15K': yago15k_embedding}


for source_db in source_dbs:
    source_ent2id = {}
    source_id2ent = {}
    source_rel2id = {}
    source_id2rel = {}
    target_ent2id = {}
    target_id2ent = {}
    target_rel2id = {}
    target_id2rel = {}
    img_embeddings = []
    source_image_embedding = source_image_embeddings[source_db]
    with open(pjoin('mmkb', source_db, 'data2', 'ent_ids'), 'r') as f1, open(pjoin('mmkb', source_db, 'data2', 'rel_ids'), 'r') as f2:
        for line in f1:
            line = line.strip()
            ent_id, ent, img_tag = line.split('\t')
            source_ent2id[ent] = int(ent_id)
            source_id2ent[int(ent_id)] = ent
            if img_tag == 'NoImage':
                img_embeddings.append(np.zeros((1, 4096)))
            else:
                img_embeddings.append(source_image_embedding[img_tag])
            
        
        for line in f2:
            line = line.strip()
            rel_id, rel = line.split('\t')
            source_rel2id[rel] = int(rel_id)
            source_id2rel[int(rel_id)] = rel
    
    with open(pjoin('mmkb', 'FB15K', 'data2', 'ent_ids'), 'r') as f1, open(pjoin('mmkb', 'FB15K', 'data2', 'rel_ids'), 'r') as f2:
        for line in f1:
            line = line.strip()
            ent_id, ent, img_tag = line.split('\t')
            target_ent2id[ent] = int(ent_id)
            target_id2ent[int(ent_id)] = ent
            if img_tag == 'NoImage':
                img_embeddings.append(np.zeros((1, 4096)))
            else:
                img_embeddings.append(np.array(fb15k_embedding[img_tag]))
        
        for line in f2:
            line = line.strip()
            rel_id, rel = line.split('\t')
            target_rel2id[rel] = int(rel_id)
            target_id2rel[int(rel_id)] = rel
    
    # concat image embeddings (all the image embedding shape is (1, 4096)), final shape is (n, 4096)
    print(set([img_embeddings[i].shape for i in range(len(img_embeddings))]))
    img_embeddings = np.concatenate(img_embeddings, axis=0)
    print('img_embeddings.shape: ', img_embeddings.shape)
    
    # save image embeddings
    img_embeddings.dump(pjoin('data2', source_db + '-' + 'FB15K', 'concat_embedding.npy'))

    

    max_source_ent_id = max(source_ent2id.values())
    max_source_rel_id = max(source_rel2id.values())
    print('max_source_ent_id: ', max_source_ent_id)
    print('max_source_rel_id: ', max_source_rel_id)
    with open(pjoin('mmkb', source_db, source_db + '_SameAsLink.txt'), 'r') as f1, open(pjoin('data2', source_db + '-' + 'FB15K', 'ref_ent_ids'), 'w') as f2:
        for line in f1:
            target_ent, _, source_ent, __ = line.split(' ')
            source_ent_id = source_ent2id[source_ent]
            target_end_id = max_source_ent_id + target_ent2id[target_ent] + 1
            f2.write(str(source_ent_id) + '\t' + str(target_end_id))
            f2.write('\n')
    
    with open(pjoin('mmkb', source_db, 'data2', 'triples'), 'r') as f1, open(pjoin('data2', source_db + '-' + 'FB15K', 'triples_1'), 'w') as f2:
        for line in f1:
            ent1, rel, ent2 = line.strip().split('\t')
            f2.write(str(ent1) + '\t' + str(rel) + '\t' + str(ent2))
            f2.write('\n')
    
    with open(pjoin('mmkb', 'FB15K', 'data2', 'triples'), 'r') as f1, open(pjoin('data2', source_db + '-' + 'FB15K', 'triples_2'), 'w') as f2:
        for line in f1:
            ent1, rel, ent2 = line.strip().split('\t')
            ent1 = int(ent1) + max_source_ent_id + 1
            ent2 = int(ent2) + max_source_ent_id + 1
            rel = int(rel) + max_source_rel_id + 1
            f2.write(str(ent1) + '\t' + str(rel) + '\t' + str(ent2))
            f2.write('\n')
    
    with open(pjoin('mmkb', source_db, 'data2', 'ent_ids'), 'r') as f1, open(pjoin('data2', source_db + '-' + 'FB15K', 'ent_ids_1'), 'w') as f2:
        for line in f1:
            line = line.strip()
            ent_id, ent, _ = line.split('\t')
            f2.write(str(ent_id) + '\t' + ent)
            f2.write('\n')
    

    max_entity_id = max_source_ent_id
    with open(pjoin('mmkb', 'FB15K', 'data2', 'ent_ids'), 'r') as f1, open(pjoin('data2', source_db + '-' + 'FB15K', 'ent_ids_2'), 'w') as f2:
        for line in f1:
            line = line.strip()
            ent_id, ent, _ = line.split('\t')
            ent_id = int(ent_id) + max_source_ent_id + 1
            max_entity_id = max(max_entity_id, ent_id)
            f2.write(str(ent_id) + '\t' + ent)
            f2.write('\n')
    
    with open(pjoin('mmkb', source_db, 'data2', 'rel_ids'), 'r') as f1, open(pjoin('data2', source_db + '-' + 'FB15K', 'rel_ids1'), 'w') as f2:
        for line in f1:
            rel_id, rel = line.strip().split('\t')
            f2.write(str(rel_id) + '\t' + str(rel))
            f2.write('\n')
    
    max_relation_id = max_source_rel_id
    print('max_entity_id: ', max_entity_id)
    print('max_relation_id: ', max_relation_id)
    with open(pjoin('mmkb', 'FB15K', 'data2', 'rel_ids'), 'r') as f1, open(pjoin('data2', source_db + '-' + 'FB15K', 'rel_ids2'), 'w') as f2:
        for line in f1:
            rel_id, rel = line.strip().split('\t')
            rel_id = int(rel_id) + max_source_rel_id + 1
            max_relation_id = max(max_relation_id, rel_id)
            max_relation_id = max(max_relation_id, rel_id)
            f2.write(str(rel_id) + '\t' + str(rel))
            f2.write('\n')
            


    # # start to establish the image entity and (entity <--> image) relation
    # source_image_relation_id = max_relation_id + 1
    # source_image_entity_id = max_entity_id + 1
    # source_image_datast = image_datasts[source_db]
    

    # with open(pjoin('data2', source_db + '-' + 'FB15K', 'triples_1'), 'a') as f:
    #     for i in range(len(source_image_datast)):
    #         entity_id, entity_name = source_image_datast[i]
    #         entity_id = int(entity_id)
    #         if entity_id not in source_id2ent:
    #             print('entity id not found: ', entity_id)
    #             continue
    #         f.write(str(entity_id) + '\t' + str(source_image_relation_id) + '\t' + str(source_image_entity_id))
    #         f.write('\n')
    #         source_image_entity_id += 1
    
    # with open(pjoin('data2', source_db + '-' + 'FB15K', 'triples_2'), 'a') as f:
    #     for i in range(len(fb15k_image_dataset)):
    #         entity_id, img, entity_name = fb15k_image_dataset[i]
    #         entity_id = int(entity_id)
    #         if entity_id not in target_id2ent:
    #             print('entity id not found: ', entity_id)
    #             continue
    #         entity_id += max_source_ent_id + 1
    #         f.write(str(entity_id) + '\t' + str(source_image_relation_id) + '\t' + str(source_image_entity_id))
    #         f.write('\n')
    #         source_image_entity_id += 1