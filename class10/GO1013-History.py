# GO1013-History
# Executa o treino do modelo com mini-batches de 128 amostras por até 50 épocas,
# usando o conjunto de validação e os callbacks para controle automático do processo.
# history = model.fit(
#     X_train, y_train,
#     batch_size=128,
#     epochs=50,
#     validation_data=(X_val, y_val),
#     callbacks=callbacks,
#     verbose=1
# )

if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten

    # Constrói modelo MNIST com split de validação e treina para capturar histórico
    (X_train, y_train), _ = keras.datasets.mnist.load_data()
    X_train = X_train / 255.0

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        batch_size=128,
                        epochs=10,
                        validation_split=0.2,
                        verbose=0)

    # Gráfico das curvas de loss e accuracy extraídas do histórico
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['loss'], label='Treino')
    axes[0].plot(history.history['val_loss'], label='Validação')
    axes[0].set_title('Loss por Época')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot(history.history['accuracy'], label='Treino')
    axes[1].plot(history.history['val_accuracy'], label='Validação')
    axes[1].set_title('Accuracy por Época')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout()
    plt.show()
