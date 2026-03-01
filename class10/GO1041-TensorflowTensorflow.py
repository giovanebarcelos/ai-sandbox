# GO1041-TensorflowTensorflow
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# Carregar (limitar vocabulário)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# Padding (mesmo tamanho)
X_train = sequence.pad_sequences(X_train, maxlen=200)
X_test = sequence.pad_sequences(X_test, maxlen=200)
