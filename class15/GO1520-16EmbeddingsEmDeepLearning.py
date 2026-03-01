# GO1520-16EmbeddingsEmDeepLearning
from tensorflow.keras.layers import Embedding

embedding_layer = Embedding(
    input_dim=vocab_size,
    output_dim=128,
    input_length=max_len
)

embedding_matrix = np.zeros((vocab_size, 100))
for word, idx in word_index.items():
    if word in glove:
        embedding_matrix[idx] = glove[word]

embedding_layer = Embedding(
    vocab_size, 100,
    weights=[embedding_matrix],
    trainable=False
)
