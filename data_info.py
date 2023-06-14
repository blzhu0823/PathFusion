file_name = 'data/ja_en/triples_1'
triples = []
entity = set()
rel = set()
for line in open(file_name, 'r'):
    head, r, tail = [int(item) for item in line.split()]
    entity.add(head); entity.add(tail); rel.add(r+1)
    triples.append((head, r+1, tail))

print(min(entity), max(entity))
print(min(rel), max(rel))

file_name2 = 'data/ja_en/triples_2'
triples2 = []
entity2 = set()
rel2 = set()
for line in open(file_name2, 'r'):
    head, r, tail = [int(item) for item in line.split()]
    entity2.add(head); entity2.add(tail); rel2.add(r+1)
    triples2.append((head, r+1, tail))

print(min(entity2), max(entity2))
print(min(rel2), max(rel2))

print(entity.intersection(entity2))
print(len(rel2))