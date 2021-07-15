# get_ipython().getoutput("pip install gensim")

from gensim.models import Word2Vec, KeyedVectors
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


sentences = [
  'this is the first sentence for Word2Vec',
  'This is the second sentence',
  'yet another sentence',
  'More more sentence',
  'and the final sentence'
]

tokens = [sentence.lower().split(' ') for sentence in sentences]


# Create model
model = Word2Vec(sentences = tokens, min_count = 1)
print('Model:', model)

# Vocab
vocab = list(model.wv.index_to_key)
print('Vocab:', vocab)
print(model.wv['sentence'])


drive_dir = '/content/drive/MyDrive/NLP-In-Depth/Develop Deep Learning Models for Natural Language in Python/Data'

# Save model
model.save('/word2vec.model.bin')

# Load model
model = Word2Vec.load('/word2vec.model.bin')


X = model.wv[model.wv.index_to_key]

pca = PCA(n_components = 2)
result = pca.fit_transform(X)

plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(vocab):
    plt.annotate(word, xy = (result[i, 0], result[i, 1]))
plt.show()


filename = '../../../Documents/GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary = True)


result = model.most_similar(positive = ['king', 'woman'], negative = ['man'], topn = 1)
print(result)



