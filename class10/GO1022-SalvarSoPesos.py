# GO1022-SalvarSóPesos
# Mostra como salvar apenas os pesos do modelo (arquivo mais leve) e como restaurá-los,
# exigindo que a arquitetura seja recriada manualmente antes de carregar os pesos.
# Salvar só pesos (mais leve)
# model.save_weights('model_weights.weights.h5')

# Carregar pesos (precisa criar arquitetura antes!)
# model = create_model()  # Recriar arquitetura
# model.load_weights('model_weights.weights.h5')

if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten

    # Treina modelo MNIST e salva somente os pesos, depois recarrega em nova instância
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train / 255.0
    X_test  = X_test / 255.0

    def create_model():
        m = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
        m.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
        return m

    model = create_model()
    model.fit(X_train, y_train, epochs=3, validation_split=0.1, verbose=0)

    _, acc_before = model.evaluate(X_test, y_test, verbose=0)
    model.save_weights('model_weights.weights.h5')

    # Recarrega a arquitetura e os pesos salvos para verificar equivalência
    model2 = create_model()
    model2.load_weights('model_weights.weights.h5')
    _, acc_after = model2.evaluate(X_test, y_test, verbose=0)

    print(f"Accuracy antes: {acc_before:.4f}  |  Após carregar pesos: {acc_after:.4f}")

    # Gráfico de barras comparando accuracy antes e depois de salvar/carregar pesos
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(['Antes de salvar', 'Após carregar pesos'],
                  [acc_before, acc_after],
                  color=['#3498DB', '#E67E22'])
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, [acc_before, acc_after]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.4f}', ha='center', va='bottom')
    ax.set_title('Accuracy — Salvar e Carregar Só os Pesos')
    ax.set_ylabel('Accuracy')
    plt.tight_layout()
    plt.show()
