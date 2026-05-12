# GO1012-Callbacks
# Configura três callbacks essenciais para treino robusto: EarlyStopping para parar
# sem melhora, ModelCheckpoint para salvar o melhor modelo e ReduceLROnPlateau
# para reduzir o learning rate automaticamente quando a val_loss estacionar.
# callbacks = [
#     EarlyStopping(
#         monitor='val_loss',
#         patience=10,
#         restore_best_weights=True,
#         verbose=1
#     ),
#     ModelCheckpoint(
#         'mnist_best.keras',
#         monitor='val_accuracy',
#         save_best_only=True,
#         mode='max',
#         verbose=1
#     ),
#     ReduceLROnPlateau(
#         monitor='val_loss',
#         factor=0.5,
#         patience=5,
#         min_lr=1e-7,
#         verbose=1
#     )
# ]

if __name__ == "__main__":
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    # Constrói modelo MNIST e treina com os três callbacks configurados
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train / 255.0

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5,
                      restore_best_weights=True, verbose=0),
        ModelCheckpoint('mnist_best.keras', monitor='val_accuracy',
                        save_best_only=True, mode='max', verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=1e-7, verbose=0)
    ]

    history = model.fit(X_train, y_train, epochs=20,
                        validation_split=0.1,
                        callbacks=callbacks, verbose=0)

    val_loss = history.history['val_loss']
    best_epoch = int(np.argmin(val_loss))

    # Gráfico de curvas loss + val_loss com linha vertical na melhor época
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Treino loss')
    plt.plot(val_loss, label='Val loss')
    plt.axvline(x=best_epoch, color='red', linestyle='--',
                label=f'Melhor época ({best_epoch})')
    plt.title('Loss com Callbacks (EarlyStopping + ReduceLROnPlateau)')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
