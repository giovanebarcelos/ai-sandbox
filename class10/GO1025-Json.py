# GO1025-Json
# Persiste o histórico de treino (loss, accuracy por época) em JSON para
# permitir visualizações futuras sem precisar retreinar o modelo.
import json

# Salvar histórico
# with open('history.json', 'w') as f:
#     json.dump(history.history, f)

# Carregar histórico
# with open('history.json', 'r') as f:
#     history_loaded = json.load(f)

# Plotar novamente
# plt.plot(history_loaded['accuracy'])

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

if __name__ == "__main__":

    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten

    # Treina modelo MNIST, salva o histórico em JSON e recarrega para plotar
    (X_train, y_train), _ = keras.datasets.mnist.load_data()
    X_train = X_train / 255.0

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=5,
                        validation_split=0.1, verbose=0)

    # Persiste e recarrega o histórico em JSON para demonstrar o ciclo completo
    with open('GO1025_history.json', 'w') as f:
        json.dump(history.history, f)
    with open('GO1025_history.json', 'r') as f:
        history_loaded = json.load(f)

    # Gráfico das curvas de accuracy carregadas do arquivo JSON
    plt.figure(figsize=(7, 4))
    plt.plot(history_loaded['accuracy'], label='Treino (do JSON)')
    plt.plot(history_loaded['val_accuracy'], label='Validação (do JSON)')
    plt.title('Curvas de Accuracy — Histórico carregado do JSON')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()
