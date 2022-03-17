from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import json
from tqdm import tqdm
import random


common_texts = []
with open('./corpus_add_8.json','r') as f:
    for i in tqdm(f):
        item = json.loads(i)
        cops = item['corpus']
        for cop in cops:
            tags = [cop['type'] + '_' + t_ for t_ in cop['tag']]
            common_texts.append((cop['text'],tags))

print(len(common_texts))

random.shuffle(common_texts)

print('starting making TaggedDocument')
documents = [TaggedDocument(doc, tag_list) for doc, tag_list in common_texts]

print('starting training dm1')
model = Doc2Vec(documents, vector_size=128, window=5, min_count=8, workers=16, negative=5, dm=1, sample=1e-6)
fname = "./doc2vec_model_dm1_8_16"
model.save(fname)

print('starting training dm1 6')
model = Doc2Vec(documents, vector_size=128, window=5, min_count=8, workers=16, negative=5, dm=1, sample=1e-6)
fname = "./doc2vec_model_dm1_add_16_4"
model.save(fname)

print('starting training dm0')
model = Doc2Vec(documents, vector_size=128, window=5, min_count=8, workers=16, negative=5, dm=0, dbow_words=1, sample=1e-5)
fname = "./doc2vec_model_dm0_8"
model.save(fname)

