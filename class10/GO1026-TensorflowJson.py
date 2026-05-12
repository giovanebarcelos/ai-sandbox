# GO1026-TensorflowJson
# Pipeline de treino com persistência completa: salva o melhor modelo via checkpoint,
# armazena o histórico em JSON e demonstra como recarregar o modelo para predição.
from tensorflow import keras
import json

# 1. Treinar com checkpoint
# checkpoint = ModelCheckpoint('best.keras', save_best_only=True)
# history = model.fit(X_train, y_train, callbacks=[checkpoint])

# 2. Salvar histórico
# with open('history.json', 'w') as f:
#     json.dump(history.history, f)

# 3. Mais tarde... carregar modelo
# model = keras.models.load_model('best.keras')

# 4. Usar para predição
# predictions = model.predict(new_data)

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
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow.keras.callbacks import ModelCheckpoint

    # Treina pipeline completo: checkpoint + JSON, recarrega e plota
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train / 255.0
    X_test  = X_test / 255.0

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    checkpoint = ModelCheckpoint('GO1026_best.keras', save_best_only=True,
                                 monitor='val_accuracy', mode='max', verbose=0)
    history = model.fit(X_train, y_train, epochs=5,
                        validation_split=0.1,
                        callbacks=[checkpoint], verbose=0)

    with open('GO1026_history.json', 'w') as f:
        json.dump(history.history, f)
    with open('GO1026_history.json', 'r') as f:
        history_loaded = json.load(f)

    loaded_model = keras.models.load_model('GO1026_best.keras')
    new_data = X_test[:5]
    predictions = loaded_model.predict(new_data, verbose=0)

    # Gráfico das curvas de loss e accuracy do pipeline completo
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_loaded['loss'], label='Treino')
    axes[0].plot(history_loaded['val_loss'], label='Validação')
    axes[0].set_title('Loss — Pipeline Completo')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot(history_loaded['accuracy'], label='Treino')
    axes[1].plot(history_loaded['val_accuracy'], label='Validação')
    axes[1].set_title('Accuracy — Pipeline Completo')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout()
    plt.show()
