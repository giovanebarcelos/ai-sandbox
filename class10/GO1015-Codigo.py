# GO1015-Codigo
# Plota lado a lado as curvas de loss e acurácia (treino vs. validação) ao longo
# das épocas, com linha vertical indicando a melhor época salva pelo checkpoint.

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

if __name__ == "__main__":
    # Carregar e normalizar MNIST para ter um history real de treino
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('GO1015_best.keras', monitor='val_accuracy',
                        save_best_only=True, mode='max')
    ]

    history = model.fit(X_train, y_train, epochs=20, batch_size=128,
                        validation_split=0.1, callbacks=callbacks, verbose=0)

    # Determinar a melhor época pelo val_accuracy
    best_epoch = int(history.history['val_accuracy'].index(
                     max(history.history['val_accuracy'])))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.axvline(best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy
    ax2.plot(history.history['accuracy'], label='Train Acc')
    ax2.plot(history.history['val_accuracy'], label='Val Acc')
    ax2.axvline(best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
