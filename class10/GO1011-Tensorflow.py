# GO1011-Tensorflow
# Compila o modelo com o otimizador Adam de learning rate explícito, loss para
# classificação multiclasse inteira e métrica de acurácia.
from tensorflow.keras.optimizers import Adam

# model.compile(
#     optimizer=Adam(learning_rate=0.001),
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

if __name__ == "__main__":
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow import keras

    # Constrói modelo simples, gera dados sintéticos e treina para demonstrar compile()
    (X_train, y_train), _ = keras.datasets.mnist.load_data()
    X_train = X_train[:2000] / 255.0
    y_train = y_train[:2000]

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(X_train, y_train, epochs=10,
                        validation_split=0.2, verbose=0)

    # Gráfico da curva de accuracy ao longo das épocas
    plt.figure(figsize=(7, 4))
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title('Curva de Accuracy')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()
