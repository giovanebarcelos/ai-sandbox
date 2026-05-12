# GO1032-Tensorflow
# Implementa um bloco residual (skip connection): a saída de duas camadas Dense é
# somada à entrada original com Add(), permitindo que gradientes fluam diretamente
# e facilitando o treinamento de redes mais profundas.
from tensorflow.keras.layers import Add

# inputs = Input(shape=(784,))
# x = Dense(128, 'relu')(inputs)
#
# # Bloco residual
# residual = x
# x = Dense(128, 'relu')(x)
# x = Dense(128, 'relu')(x)
# x = Add()([x, residual])  # Skip connection
#
# outputs = Dense(10, 'softmax')(x)
# model = Model(inputs, outputs)

if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    from tensorflow import keras
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import Dense, Flatten

    # Constrói modelo com skip connection (bloco residual) e treina no MNIST
    (X_train, y_train), _ = keras.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 784).astype('float32') / 255.0

    inputs  = Input(shape=(784,))
    x       = Dense(128, activation='relu')(inputs)
    residual = x
    x       = Dense(128, activation='relu')(x)
    x       = Dense(128, activation='relu')(x)
    x       = Add()([x, residual])
    outputs = Dense(10, activation='softmax')(x)
    model   = Model(inputs, outputs)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=5,
                        validation_split=0.1, verbose=0)

    # Gráfico das curvas de loss e accuracy do modelo residual
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['loss'], label='Treino')
    axes[0].plot(history.history['val_loss'], label='Validação')
    axes[0].set_title('Loss — Bloco Residual')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot(history.history['accuracy'], label='Treino')
    axes[1].plot(history.history['val_accuracy'], label='Validação')
    axes[1].set_title('Accuracy — Bloco Residual')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout()
    plt.show()
