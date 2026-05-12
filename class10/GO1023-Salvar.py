# GO1023-Salvar
# Demonstra o formato SavedModel do TensorFlow: salva o modelo como diretório
# com arquivos .pb e variáveis separados, e como recarregá-lo posteriormente.
# Salvar (cria diretório)
# model.save('saved_model/')

# Estrutura criada:
# saved_model/
# ├── saved_model.pb
# ├── variables/
# │   ├── variables.data-00000-of-00001
# │   └── variables.index
# └── assets/

# Carregar
# model = keras.models.load_model('saved_model/')

if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten

    # Treina modelo, salva no formato SavedModel (diretório) e recarrega para avaliação
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
    model.fit(X_train, y_train, epochs=3, validation_split=0.1, verbose=0)

    _, acc_before = model.evaluate(X_test, y_test, verbose=0)
    model.save('GO1023_saved_model')

    # Recarrega o SavedModel do diretório e avalia para confirmar equivalência
    loaded_model = keras.models.load_model('GO1023_saved_model')
    _, acc_after = loaded_model.evaluate(X_test, y_test, verbose=0)

    print(f"Accuracy antes: {acc_before:.4f}  |  Após carregar: {acc_after:.4f}")

    # Gráfico de barras comparando accuracy antes e depois de salvar com SavedModel
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(['Antes de salvar', 'Após carregar (SavedModel)'],
                  [acc_before, acc_after],
                  color=['#8E44AD', '#1ABC9C'])
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, [acc_before, acc_after]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.4f}', ha='center', va='bottom')
    ax.set_title('Accuracy — Formato SavedModel')
    ax.set_ylabel('Accuracy')
    plt.tight_layout()
    plt.show()
