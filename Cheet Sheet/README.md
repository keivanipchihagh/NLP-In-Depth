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

- Train & Test Split | sklearn
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)
```
- LabelEncoder | sklearn
```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoded_x = encoder.fit_transform(x)
mapper = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
```
- Text to Word Sequence | Keras
```python
from keras.preprocessing.text import text_to_word_sequence
result = text_to_word_sequence(text)
```
- One Hot | Keras
```python
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import text_to_word_sequence
words = set(text_to_word_sequence(text))
vocab_size = len(words)
result = one_hot(text, round(vocab_size * 1.3))
```
- Hashing Trick | Keras
```python
from keras.preprocessing.text import hashing_trick
from keras.preprocessing.text import text_to_word_sequence
words = set(text_to_word_sequence(text))
vocab_size = len(words)
result = hashing_trick(text, round(vocab_size * 1.2), hash_function = 'md5')
```
