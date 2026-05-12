# GO1007-TensorflowTensorflow
# Demonstra a API Funcional do Keras: define o fluxo de dados passando tensores entre
# camadas e cria o modelo a partir de inputs e outputs explícitos com Model().
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense

inputs = Input(shape=(784,))
x = Dense(64, activation='relu')(inputs)
x = Dense(32, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tensorflow import keras

    # Carrega MNIST, compila e treina o modelo Functional API por 5 épocas
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
    X_test  = X_test.reshape(-1, 784).astype('float32') / 255.0

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=5,
                        validation_split=0.1, verbose=1)

    # Gráfico de curvas de loss e accuracy ao longo das épocas
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['loss'], label='Treino')
    axes[0].plot(history.history['val_loss'], label='Validação')
    axes[0].set_title('Loss por Época (Functional API)')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot(history.history['accuracy'], label='Treino')
    axes[1].plot(history.history['val_accuracy'], label='Validação')
    axes[1].set_title('Accuracy por Época (Functional API)')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('GO1007-history.png', dpi=100, bbox_inches='tight')
    plt.close()
