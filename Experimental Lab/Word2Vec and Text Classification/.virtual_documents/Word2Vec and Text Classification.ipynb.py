from gensim.models import KeyedVectors

# Load Word2Vec Model
word2vec_model_path = '../../../../Documents/Word2Vec/GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary = True)


hello_vectors = model['hello']
print(f'Hello {len(hello_vectors)} Dimention Vector:\n', hello_vectors)


import numpy as np

cat = model['cat']
dog = model['dog']

dist = np.linalg.norm(cat - dog)
print(f'Distance between can and dog: {dist}')


model.most_similar(positive = ['woman', 'king'], negative = ['man'])


model.doesnt_match("house garage store dog".split())


model.similarity('iphone', 'android')


model.most_similar('cat')
