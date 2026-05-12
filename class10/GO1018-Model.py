# GO1018-Model
# Ilustra a necessidade de incluir a camada Flatten antes das camadas Dense
# ao trabalhar com imagens 2D do MNIST no modelo Sequential.
# model = Sequential([
#     Flatten(input_shape=(28, 28)),  # ← Adicionar esta linha!
#     Dense(128, 'relu'),
#     ...
# ])

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

    # Constrói modelo correto com Flatten e treina no MNIST por 5 épocas
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
    history = model.fit(X_train, y_train, epochs=5,
                        validation_split=0.1, verbose=0)

    # Gráfico das curvas de loss mostrando convergência com Flatten corretamente incluído
    plt.figure(figsize=(7, 4))
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Loss — Modelo com Flatten (correto)')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
