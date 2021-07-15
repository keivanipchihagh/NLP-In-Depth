# NLP-In-Depth Cheetsheet

## Keras
- One-Hot-Encoding
```python
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words = 1000, lower = True, split = ' ', char_level = False, oov_token = None)
tokenizer.fit_on_texts(sentences)
one_hot_encodes = tokenizer.texts_to_matrix(sentences, mode = 'binary') # Modes: 'binary', 'count', 'freq', 'tfidf'
```
- Plot Model
```python
from keras.utils.vis_utils import plot_model
plot_model(model = model, to_file = 'path_to_file.png', show_shapes = False, show_dtypes = False, show_layer_names = True, rankdir = 'TB', expand_nested = False, dpi = 96)
```
- Pad Sequence
```python
from keras.preprocessing.sequence import pad_sequences
X = pad_sequences(X, maxlen = maxlen, dtype = 'int32', padding = 'pre', truncating = 'pre', value = 0.0) # padding & truncating: 'pre', 'post'
```
