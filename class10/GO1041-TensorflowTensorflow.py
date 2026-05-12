# GO1041-TensorflowTensorflow
# Carrega o dataset IMDB de críticas de filmes, limitando o vocabulário a 10 000 palavras,
# e aplica padding para que todas as sequências tenham o mesmo comprimento (200 tokens).
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# Carregar (limitar vocabulário)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# Padding (mesmo tamanho)
X_train = sequence.pad_sequences(X_train, maxlen=200)
X_test = sequence.pad_sequences(X_test, maxlen=200)

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

if __name__ == "__main__":

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D

    # Constrói modelo com Embedding + GlobalAveragePooling para classificação de sentimento
    model = Sequential([
        Embedding(input_dim=10000, output_dim=16, input_length=200),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=3,
                        batch_size=512,
                        validation_split=0.1, verbose=1)

    # Gráfico das curvas de loss e accuracy no dataset IMDB
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['loss'], label='Treino')
    axes[0].plot(history.history['val_loss'], label='Validação')
    axes[0].set_title('Loss — IMDB Sentiment')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot(history.history['accuracy'], label='Treino')
    axes[1].plot(history.history['val_accuracy'], label='Validação')
    axes[1].set_title('Accuracy — IMDB Sentiment')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout()
    plt.show()
