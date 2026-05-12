# GO1014-Codigo
# Avalia o modelo treinado no conjunto de teste e exibe a acurácia final,
# representando o desempenho real esperado em dados nunca vistos.
# test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
# print(f"\n🎉 Test Accuracy: {test_acc*100:.2f}%")
# 🎉 Test Accuracy: 99.23%

if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten

    # Constrói, treina e avalia modelo MNIST no conjunto de teste
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train / 255.0
    X_test  = X_test / 255.0

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, validation_split=0.1, verbose=0)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")

    # Gráfico de barras mostrando loss e accuracy obtidos no conjunto de teste
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(['Test Loss', 'Test Accuracy'], [test_loss, test_acc],
                  color=['#E74C3C', '#2ECC71'])
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, [test_loss, test_acc]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.4f}', ha='center', va='bottom')
    ax.set_title('Avaliação no Conjunto de Teste')
    ax.set_ylabel('Valor')
    plt.tight_layout()
    plt.show()
