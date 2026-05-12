# GO1019-Reduzir
# Lista de boas práticas de otimização: reduzir o learning rate, normalizar as
# entradas e usar inicialização adequada para estabilizar o treinamento.
# optimizer=Adam(learning_rate=0.0001)  # Reduzir LR
# X_train = X_train / 255.0             # Normalizar
# kernel_initializer='he_normal'        # Boa inicialização

if __name__ == "__main__":
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow.keras.optimizers import Adam

    # Carrega MNIST e treina dois modelos: com normalização (LR padrão) e sem (LR alto)
    (X_train_raw, y_train), _ = keras.datasets.mnist.load_data()
    X_norm   = (X_train_raw[:3000] / 255.0)
    X_nonorm = X_train_raw[:3000].reshape(3000, -1).astype('float32')
    y        = y_train[:3000]

    def build_model(lr):
        m = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(64, activation='relu', kernel_initializer='he_normal'),
            Dense(10, activation='softmax')
        ])
        m.compile(optimizer=Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
        return m

    hist_norm   = build_model(lr=0.0001).fit(
        X_norm, y, epochs=15, validation_split=0.2, verbose=0)
    hist_nonorm = build_model(lr=0.01).fit(
        X_nonorm.reshape(-1, 28, 28), y, epochs=15,
        validation_split=0.2, verbose=0)

    # Gráfico comparativo das curvas de loss: normalizado vs não normalizado
    plt.figure(figsize=(8, 4))
    plt.plot(hist_norm.history['loss'],   label='Normalizado (lr=1e-4)')
    plt.plot(hist_nonorm.history['loss'], label='Sem normalização (lr=1e-2)', linestyle='--')
    plt.title('Comparação de Loss: Normalização e LR')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
