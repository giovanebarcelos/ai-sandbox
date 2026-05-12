# GO1048-KerasFazIsso
# Lembra que o Keras gerencia automaticamente o modo de inferência (training=False)
# ao chamar model.predict(), aplicando corretamente Dropout e BatchNormalization.
# Keras faz isso automaticamente!
# model.predict(X_test)

if __name__ == "__main__":
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten

    # Constrói e treina modelo MNIST, depois realiza predict em 10 amostras do teste
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train / 255.0
    X_test  = X_test / 255.0

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=3, validation_split=0.1, verbose=0)

    # Keras faz isso automaticamente!
    predictions = model.predict(X_test[:10], verbose=0)
    print("Predições (primeiras 10 amostras):")
    print(np.argmax(predictions, axis=1))

    # Gráfico de barras de probabilidade para as 10 classes da primeira predição
    classes = list(range(10))
    probs   = predictions[0]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(classes, probs, color='#3498DB')
    best_cls = int(np.argmax(probs))
    bars[best_cls].set_color('#E74C3C')
    plt.xticks(classes)
    plt.title(f'Probabilidades por Classe — Amostra 0 (Predição: {best_cls})')
    plt.xlabel('Classe (dígito)')
    plt.ylabel('Probabilidade')
    plt.tight_layout()
    plt.savefig('GO1048-predict-proba.png', dpi=100, bbox_inches='tight')
    plt.close()
