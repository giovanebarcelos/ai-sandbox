# GO1249-TensorflowTensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Carregar dados


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocessar
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Criar modelo
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    # Compilar
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Treinar
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.1
    )

    # Avaliar
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc:.4f}')
    # Resultado esperado: ~99.2% accuracy

    # ─── VISUALIZAÇÃO: CURVAS DE TREINAMENTO ───
    import matplotlib
    import matplotlib.pyplot as plt

    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
    except NameError:
        pass

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2, color='#4e79a7')
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2, color='#e15759', linestyle='--')
    axes[0].set_title('Loss por Época', fontsize=13)
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss (Categorical Crossentropy)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2, color='#4e79a7')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2, color='#e15759', linestyle='--')
    axes[1].axhline(y=0.99, color='green', linestyle=':', linewidth=1.5, label='Meta 99%')
    axes[1].set_title('Acurácia por Época', fontsize=13)
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Predições em imagens de teste
    predictions = model.predict(x_test[:10], verbose=0)
    pred_labels = predictions.argmax(axis=1)
    true_labels = y_test[:10].argmax(axis=1) if len(y_test.shape) > 1 else y_test[:10]

    axes[2].bar(range(10), predictions[0], color=['#59a14f' if i == pred_labels[0] else '#cccccc'
                                                   for i in range(10)], edgecolor='black')
    axes[2].set_xticks(range(10))
    axes[2].set_xticklabels([str(i) for i in range(10)])
    axes[2].set_title(f'Distribuição de Probabilidade\n(1ª amostra | Pred: {pred_labels[0]}, Real: {true_labels[0]})', fontsize=11)
    axes[2].set_xlabel('Dígito')
    axes[2].set_ylabel('Probabilidade')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'CNN MNIST — Acurácia Final: {test_acc:.4f}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
