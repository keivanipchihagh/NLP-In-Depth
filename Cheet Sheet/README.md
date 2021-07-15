# NLP-In-Depth Cheetsheet

## Keras
- One-Hot-Encoding
```python
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words = 1000, lower = True, split = ' ', char_level = False, oov_token = None)
tokenizer.fit_on_texts(sentences)
one_hot_encodes = tokenizer.texts_to_matrix(sentences, mode = 'binary') # Modes: 'binary', 'count', 'freq', 'tfidf'
```
