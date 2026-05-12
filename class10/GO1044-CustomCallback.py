# GO1044-CustomCallback
class CustomCallback(keras.callbacks.Callback):
    # Ao final de cada época, salva o modelo a cada 5 épocas e interrompe o treino
    # imediatamente se a loss se tornar NaN, evitando desperdício de recursos.
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            self.model.save(f'model_epoch_{epoch}.keras')
        if logs.get('loss') != logs.get('loss'):  # NaN check
            self.model.stop_training = True

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

if __name__ == '__main__':
    import numpy as np
    from tensorflow import keras

    print("=== Demonstração do CustomCallback ===")

    # Modelo simples
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(4,)),
        keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Dados sintéticos
    X = np.random.rand(100, 4).astype(np.float32)
    y = np.random.randint(0, 3, 100)

    print("Treinando com CustomCallback (salva a cada 5 épocas)...")
    history = model.fit(X, y, epochs=11, callbacks=[CustomCallback()], verbose=1)

    # Gráfico das curvas de loss e accuracy registradas pelo CustomCallback

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['loss'], marker='o', markersize=4, label='Loss')
    axes[0].set_title('Loss por Época — CustomCallback')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot(history.history['accuracy'], marker='o', markersize=4,
                 color='#2ECC71', label='Accuracy')
    axes[1].set_title('Accuracy por Época — CustomCallback')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout()
    plt.show()
