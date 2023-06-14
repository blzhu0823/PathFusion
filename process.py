import os
import numpy as np
from os.path import join as pjoin
from mmkb.DB15K.image_processing2 import mydataset as db15k_image_dataset
from mmkb.YAGO15K.image_processing2 import mydataset as yago15k_image_dataset
from mmkb.FB15K.image_processing import mydataset as fb15k_image_dataset



source_dbs = ['DB15K', 'YAGO15K']
image_datasts = {'DB15K': db15k_image_dataset, 'YAGO15K': yago15k_image_dataset}


for source_db in source_dbs:
    source_ent2id = {}
    source_id2ent = {}
    source_rel2id = {}
    source_id2rel = {}
    target_ent2id = {}
    target_id2ent = {}
    target_rel2id = {}
    target_id2rel = {}
    with open(pjoin('mmkb', source_db, 'data', 'ent_ids'), 'r') as f1, open(pjoin('mmkb', source_db, 'data', 'rel_ids'), 'r') as f2:
        for line in f1:
            line = line.strip()
            ent_id, ent = line.split('\t')
            source_ent2id[ent] = int(ent_id)
            source_id2ent[int(ent_id)] = ent
        
        for line in f2:
            line = line.strip()
            rel_id, rel = line.split('\t')
            source_rel2id[rel] = int(rel_id)
            source_id2rel[int(rel_id)] = rel
    
    with open(pjoin('mmkb', 'FB15K', 'data', 'ent_ids'), 'r') as f1, open(pjoin('mmkb', 'FB15K', 'data', 'rel_ids'), 'r') as f2:
        for line in f1:
            line = line.strip()
            ent_id, ent = line.split('\t')
            target_ent2id[ent] = int(ent_id)
            target_id2ent[int(ent_id)] = ent
        
        for line in f2:
            line = line.strip()
            rel_id, rel = line.split('\t')
            target_rel2id[rel] = int(rel_id)
            target_id2rel[int(rel_id)] = rel

    max_source_ent_id = max(source_ent2id.values())
    max_source_rel_id = max(source_rel2id.values())
    print('max_source_ent_id: ', max_source_ent_id)
    print('max_source_rel_id: ', max_source_rel_id)
    with open(pjoin('mmkb', source_db, source_db + '_SameAsLink.txt'), 'r') as f1, open(pjoin('data', source_db + '-' + 'FB15K', 'ref_ent_ids'), 'w') as f2:
        for line in f1:
            target_ent, _, source_ent, __ = line.split(' ')
            source_ent_id = source_ent2id[source_ent]
            target_end_id = max_source_ent_id + target_ent2id[target_ent] + 1
            f2.write(str(source_ent_id) + '\t' + str(target_end_id))
            f2.write('\n')
    
    with open(pjoin('mmkb', source_db, 'data', 'triples'), 'r') as f1, open(pjoin('data', source_db + '-' + 'FB15K', 'triples_1'), 'w') as f2:
        for line in f1:
            ent1, rel, ent2 = line.strip().split('\t')
            f2.write(str(ent1) + '\t' + str(rel) + '\t' + str(ent2))
            f2.write('\n')
    
    with open(pjoin('mmkb', 'FB15K', 'data', 'triples'), 'r') as f1, open(pjoin('data', source_db + '-' + 'FB15K', 'triples_2'), 'w') as f2:
        for line in f1:
            ent1, rel, ent2 = line.strip().split('\t')
            ent1 = int(ent1) + max_source_ent_id + 1
            ent2 = int(ent2) + max_source_ent_id + 1
            rel = int(rel) + max_source_rel_id + 1
            f2.write(str(ent1) + '\t' + str(rel) + '\t' + str(ent2))
            f2.write('\n')
    
    with open(pjoin('mmkb', source_db, 'data', 'ent_ids'), 'r') as f1, open(pjoin('data', source_db + '-' + 'FB15K', 'ent_ids_1'), 'w') as f2:
        for line in f1:
            line = line.strip()
            ent_id, ent = line.split('\t')
            f2.write(str(ent_id) + '\t' + ent)
            f2.write('\n')
    

    max_entity_id = max_source_ent_id
    with open(pjoin('mmkb', 'FB15K', 'data', 'ent_ids'), 'r') as f1, open(pjoin('data', source_db + '-' + 'FB15K', 'ent_ids_2'), 'w') as f2:
        for line in f1:
            line = line.strip()
            ent_id, ent = line.split('\t')
            ent_id = int(ent_id) + max_source_ent_id + 1
            max_entity_id = max(max_entity_id, ent_id)
            f2.write(str(ent_id) + '\t' + ent)
            f2.write('\n')
    
    with open(pjoin('mmkb', source_db, 'data', 'rel_ids'), 'r') as f1, open(pjoin('data', source_db + '-' + 'FB15K', 'rel_ids1'), 'w') as f2:
        for line in f1:
            rel_id, rel = line.strip().split('\t')
            f2.write(str(rel_id) + '\t' + str(rel))
            f2.write('\n')
    
    max_relation_id = max_source_rel_id
    print('max_entity_id: ', max_entity_id)
    print('max_relation_id: ', max_relation_id)
    with open(pjoin('mmkb', 'FB15K', 'data', 'rel_ids'), 'r') as f1, open(pjoin('data', source_db + '-' + 'FB15K', 'rel_ids2'), 'w') as f2:
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
    

    # with open(pjoin('data', source_db + '-' + 'FB15K', 'triples_1'), 'a') as f:
    #     for i in range(len(source_image_datast)):
    #         entity_id, entity_name = source_image_datast[i]
    #         entity_id = int(entity_id)
    #         if entity_id not in source_id2ent:
    #             print('entity id not found: ', entity_id)
    #             continue
    #         f.write(str(entity_id) + '\t' + str(source_image_relation_id) + '\t' + str(source_image_entity_id))
    #         f.write('\n')
    #         source_image_entity_id += 1
    
    # with open(pjoin('data', source_db + '-' + 'FB15K', 'triples_2'), 'a') as f:
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