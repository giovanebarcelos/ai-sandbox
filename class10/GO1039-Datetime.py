# GO1039-Datetime
# Salva o modelo e seus hiperparâmetros com timestamp no nome do arquivo, permitindo
# rastrear e comparar diferentes versões de experimentos ao longo do tempo.
# Salvar com versão
# import datetime
# timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# model.save(f'model_{timestamp}.keras')

# Salvar hiperparâmetros
# import json
# config = {
#     'learning_rate': 0.001,
#     'batch_size': 128,
#     'epochs': 50,
#     'dropout': 0.3
# }
# with open(f'config_{timestamp}.json', 'w') as f:
#     json.dump(config, f)

if __name__ == "__main__":
    import datetime
    import json
    import matplotlib
    import matplotlib.pyplot as plt
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten

    # Treina modelo simples, salva com timestamp e registra os hiperparâmetros em JSON
    (X_train, y_train), _ = keras.datasets.mnist.load_data()
    X_train = X_train / 255.0

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=5,
                        validation_split=0.1, verbose=0)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model.save(f'GO1039_model_{timestamp}.keras')

    config = {'learning_rate': 0.001, 'batch_size': 128,
              'epochs': 5, 'dropout': 0.0}
    with open(f'GO1039_config_{timestamp}.json', 'w') as f:
        json.dump(config, f)

    print(f"Modelo salvo como: GO1039_model_{timestamp}.keras")

    # Gráfico das curvas de loss para ilustrar o treinamento versionado
    plt.figure(figsize=(7, 4))
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title(f'Loss — Versão {timestamp}')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
