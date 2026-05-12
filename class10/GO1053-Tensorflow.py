# GO1053-Tensorflow
# Configura o treinamento distribuído com MirroredStrategy para múltiplas GPUs:
# o modelo é criado dentro do escopo da estratégia e o Keras gerencia a sincronização.
import tensorflow as tf

# Estratégia para múltiplas GPUs
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

# Criar modelo dentro da estratégia
# with strategy.scope():
#     model = create_model()
#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])

# Treinar normalmente - Keras gerencia distribuição
# model.fit(X_train, y_train, epochs=10, batch_size=128)

if __name__ == "__main__":
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from tensorflow import keras

    # Cria modelo dentro da strategy.scope() e treina com dados sintéticos
    (X_train, y_train), _ = keras.datasets.mnist.load_data()
    X_train = X_train[:3000] / 255.0
    y_train = y_train[:3000]

    with strategy.scope():
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=5,
                        batch_size=128,
                        validation_split=0.1, verbose=0)

    # Gráfico das curvas de loss do treinamento distribuído (MirroredStrategy)
    plt.figure(figsize=(7, 4))
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title(f'Loss — MirroredStrategy ({strategy.num_replicas_in_sync} dispositivo(s))')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
