# NLP-In-Depth Cheetsheet

## Keras
- Tokenizer | keras
```python
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words = 1000, lower = True, split = ' ', char_level = False, oov_token = None)
tokenizer.fit_on_texts(sentences)
one_hot_encodes = tokenizer.texts_to_matrix(sentences, mode = 'binary') # Modes: 'binary', 'count', 'freq', 'tfidf'
```
- Plot Model | keras
```python
from keras.utils.vis_utils import plot_model
plot_model(model = model, to_file = 'path_to_file.png', show_shapes = False, show_dtypes = False, show_layer_names = True, rankdir = 'TB', expand_nested = False, dpi = 96)
```
- Pad Sequence | keras
```python
from keras.preprocessing.sequence import pad_sequences
X = pad_sequences(X, maxlen = maxlen, dtype = 'int32', padding = 'pre', truncating = 'pre', value = 0.0) # padding & truncating: 'pre', 'post'
```
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
- Hashing Trick
```python
from keras.preprocessing.text import hashing_trick
from keras.preprocessing.text import text_to_word_sequence
words = set(text_to_word_sequence(text))
vocab_size = len(words)
result = hashing_trick(text, round(vocab_size * 1.2), hash_function = 'md5')
```
