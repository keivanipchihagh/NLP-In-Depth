# NLP Cheatsheet

### keras | Preprocessing | Tokenizer
```python
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(
  num_words = 1000,       # the maximum number of words to keep (Only the most common num_words-1 words will be kept)
  lower = True,           # boolean. Whether to convert the texts to lowercase.
  split = ' ',            # str. Separator for word splitting.
  char_level = False,     # if True, every character will be treated as a token.
  oov_token = None        # replaces out-of-vocabulary words during text_to_sequence calls with oov_token
)
tokenizer.fit_on_texts(sentences)   # can be a list of strings
one_hot_encodes = tokenizer.texts_to_matrix(sentences, mode = 'binary') # Modes: 'binary', 'count', 'freq', 'tfidf'
```
More: [tensorflow documentation](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer)
<hr/>

### Keras | Utils | Plot Model
```python
from keras.utils.vis_utils import plot_model

plot_model(
  model = model,                  # a Keras model instance
  to_file = 'path_to_file.png',   # file name of the plot image.
  show_shapes = False,            # whether to display shape information.
  show_dtypes = False,            # whether to display layer dtypes.
  show_layer_names = True,        # whether to display layer names.
  rankdir = 'TB',                 # format for PyDot: 'TB' creates a vertical plot; 'LR' creates a horizontal plot.
  expand_nested = False,          # whether to expand nested models into clusters.
  dpi = 96                        # dots per inch.
)
```
More: [tensorflow decumentation](https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model)
<hr/>

### Keras | Preprocessing | Pad Sequence
```python
from keras.preprocessing.sequence import pad_sequences

X = pad_sequences(
  sequences,            # list of sequences (each sequence is a list of integers).
  maxlen = maxlen,      # maximum length of all sequences. defaults to the length of the longest individual sequence.
  dtype = 'int32',      # type of the output sequences.
  padding = 'pre',      # pad either before or after each sequence ('pre' or 'post')
  truncating = 'pre',   # remove values from sequences larger than maxlen ('pre' or 'post')
  value = 0.0           # float or String, padding value.
)
```
More: [tensorflow decumentation](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences)
<hr/>

### Scikit-Learn | Model Selection | Train & Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
  x_arr,
  y_arr,
  test_size = 0.1,    # should be in rnage (0.0,1.0) and is the proportion of the dataset to include in the test split.
  random_state = 42,  # controls the shuffling applied to the data before applying the split
  shuffle = True,     # whether or not to shuffle the data before splitting (if shuffle=False then stratify must be None)
  stratify = None
)
```
More: [scikit-learn decumentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
<hr/>

### Scikit-Learn | Preprocessing | LabelEncoder
```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoded_x = encoder.fit_transform(x)                                        # Fit label encoder and return encoded labels.
x = inverse_transform(y)                                                    # transform labels back to original encoding.
mapper = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))   # word to encode index mapper
```
More: [scikit-learn decumentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
<hr/>

- Keras | Preprocessing | One Hot
```python
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import text_to_word_sequence

words = set(text_to_word_sequence(text))            # convert string sequence to its words
vocab_size = len(words)
result = one_hot(
  input_text = text,                                  # input text (string)
  n = round(vocab_size * 1.3),                        # size of vocabulary
  filters = ''!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
  lower = True,                                       # boolean. whether to set the text to lowercase.
  split = ' '                                         # str. separator for word splitting.
)
```
More: [tensorflow decumentation](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/one_hot)
<hr/>

### Keras | Layer | Embedding
```python
from keras.layers import Embedding

embedding = Embedding(
  input_dim = vocab_size,                      # size of the vocabulary
  output_dim = <number>,                       # dimension of the dense embedding
  mask_zero = False                            # whether the value '0' is a special "padding" value that should be masked out
  input_length = max_length_of_each_sequence   # constant length of input sequences (required if Flatten and Dense are used!)
)   # returns: 3D tensor with shape: (batch_size, input_length, output_dim)
```
More: [keras decumentation](https://keras.io/api/layers/core_layers/embedding/)
<hr/>

### Keras | Layer | Pretrained Embedding with GloVe
```python
import numpy as np
from keras.layers import Embedding

# compute an index mapping words to known embeddings
embeddings_index = {}
with open(GLOVE_DIR, encoding = "utf8") as file:
  for line in f:
      values = line.split()
      embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')

# computer matrix for the vocabulary words
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector    # words not found in embedding index will be all-zeros.

# Define layer
embedding_layer = Embedding(
  input_dim = len(word_index) + 1,              # size of the vocabulary
  outpu_dim = EMBEDDING_DIM,                    # dimension of the dense embedding
  weights = [embedding_matrix],                 # weights
  input_length = MAX_SEQUENCE_LENGTH,           # constant length of input sequences
  trainable = False,                            # freezes weights so they don't change in the training process
)
```
More: [keras decumentation](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)
<hr/>
