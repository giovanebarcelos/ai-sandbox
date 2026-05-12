# GO1029-TensorflowTensorflow
# Demonstra a API Funcional do Keras com camadas Dense encadeadas de forma explícita,
# criando o modelo com Model() a partir de inputs e outputs nomeados.
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense

# Definir entrada
inputs = Input(shape=(784,))

# Camadas sequenciais
x = Dense(128, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)

# Criar modelo
model = Model(inputs=inputs, outputs=outputs, name='functional_model')
# model.compile(...)
# model.fit(...)

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

    # Treina o modelo de API Funcional no MNIST e plota as curvas de evolução
    (X_train, y_train), _ = keras.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 784).astype('float32') / 255.0

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=5,
                        validation_split=0.1, verbose=0)

    # Gráfico das curvas de loss e accuracy da Functional API
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['loss'], label='Treino')
    axes[0].plot(history.history['val_loss'], label='Validação')
    axes[0].set_title('Loss — Functional API')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot(history.history['accuracy'], label='Treino')
    axes[1].plot(history.history['val_accuracy'], label='Validação')
    axes[1].set_title('Accuracy — Functional API')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout()
    plt.show()
