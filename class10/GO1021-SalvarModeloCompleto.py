# GO1021-SalvarModeloCompleto
# Demonstra como salvar o modelo completo (arquitetura + pesos + otimizador) no
# formato .keras e recarregá-lo para uso imediato em predições.
# Salvar modelo completo
# model.save('my_model.keras')

# Carregar modelo completo
# model = keras.models.load_model('my_model.keras')

# Usar imediatamente
# predictions = model.predict(X_test)

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

if __name__ == "__main__":

    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten

    # Treina modelo MNIST, salva no formato .keras, recarrega e avalia
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
    model.save('GO1021_model.keras')

    # Recarrega o modelo salvo e avalia novamente para comparar acurácias
    loaded_model = keras.models.load_model('GO1021_model.keras')
    _, acc_after = loaded_model.evaluate(X_test, y_test, verbose=0)

    print(f"Accuracy antes de salvar: {acc_before:.4f}")
    print(f"Accuracy após carregar:   {acc_after:.4f}")

    # Gráfico de barras comparando accuracy antes e depois de salvar/carregar
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(['Antes de salvar', 'Após carregar'],
                  [acc_before, acc_after],
                  color=['#3498DB', '#2ECC71'])
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, [acc_before, acc_after]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.4f}', ha='center', va='bottom')
    ax.set_title('Accuracy — Salvar e Carregar Modelo Completo')
    ax.set_ylabel('Accuracy')
    plt.tight_layout()
    plt.show()
