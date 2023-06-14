from os.path import join as pjoin



sources = ['DB15K', 'YAGO15K']
target = 'FB15K'

for source in sources:
    with open(pjoin('data', source + '-' + target, 'attr'), 'w') as f1, open(pjoin('mmkb', source, 'data', 'attr'), 'r') as f2:
        for line in f2:
            f1.write(line)
    
    with open(pjoin('data', source + '-' + target, 'attr'), 'a') as f1, open(pjoin('mmkb', target, 'data', 'attr'), 'r') as f2:
        for line in f2:
            f1.write(line)
    

