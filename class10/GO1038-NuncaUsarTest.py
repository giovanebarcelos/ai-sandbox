# GO1038-NuncaUsarTest
# Reforça a regra de não usar o conjunto de teste durante o desenvolvimento:
# o treino e tuning devem usar apenas train + val; o teste é reservado para avaliação final.
# NUNCA usar test set durante desenvolvimento!
# Apenas no final, uma vez

# Desenvolvimento: train + val
# model.fit(X_train, y_train, validation_data=(X_val, y_val))

# Depois de tudo pronto:
# final_score = model.evaluate(X_test, y_test)

if __name__ == "__main__":
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten

    # Treina modelo usando apenas train+val; avalia no teste somente ao final
    (X_full, y_full), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_full = X_full / 255.0
    X_test = X_test / 255.0

    split = int(0.8 * len(X_full))
    X_train_dev, X_val = X_full[:split], X_full[split:]
    y_train_dev, y_val = y_full[:split], y_full[split:]

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train_dev, y_train_dev,
                        epochs=5,
                        validation_data=(X_val, y_val),
                        verbose=0)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Avaliação final no teste: loss={test_loss:.4f}, acc={test_acc:.4f}")

    # Gráfico train+val com a avaliação final no teste marcada como ponto
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history['accuracy'], label='Treino acc')
    ax.plot(history.history['val_accuracy'], label='Val acc')
    ax.scatter(len(history.history['accuracy']) - 1, test_acc,
               color='red', s=100, zorder=5,
               label=f'Teste (final) = {test_acc:.4f}')
    ax.set_title('Treino/Val + Avaliação Final no Teste')
    ax.set_xlabel('Época')
    ax.set_ylabel('Accuracy')
    ax.legend()
    plt.tight_layout()
    plt.show()
