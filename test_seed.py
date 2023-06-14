from data.utils_db import *





with open('./attr_log.txt', 'w') as f:
    for l_ent, r_ent in seeds:
        total_len = len(id2attrs[l_ent]) + len(id2attrs[r_ent])
        f.write('key name: ' + id2ent[l_ent] + ' ' + id2ent[r_ent] + '\n')
        f.write('source entity attr: ' + str(id2attrs[l_ent]) + '\n')
        f.write('target entity attr: ' + str(id2attrs[r_ent]) + '\n')
        f.write('\n')
