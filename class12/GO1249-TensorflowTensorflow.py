# GO1249-TensorflowTensorflow
# CNN completa para classificação de dígitos MNIST — pipeline passo a passo:
# 1. Carregar dados, 2. Pré-processar, 3. Criar modelo, 4. Compilar, 5. Treinar, 6. Avaliar
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Carregar dados


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # ─── PASSO 1: Carregar dados MNIST ───
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # x_train: (60000, 28, 28) uint8 — 60k imagens grayscale 28×28
    # y_train: (60000,) int — rótulos 0-9

    # ─── PASSO 2: Pré-processar ───
    # reshape: adiciona dimensão de canal (Conv2D espera 4D: batch,H,W,C)
    # /255: normaliza [0,255] → [0,1] — acelera convergência, estabiliza gradientes
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    # to_categorical: converte rótulo inteiro em vetor one-hot
    # Ex: 3 → [0,0,0,1,0,0,0,0,0,0] — necessário para categorical_crossentropy
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # ─── PASSO 3: Criar modelo CNN ───
    model = keras.Sequential([
        # Bloco 1: conv+pool — input 28×28 → 13×13
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),  # (26×26×32)
        layers.MaxPooling2D((2,2)),    # reduz pela metade: (13×13×32)
        # Bloco 2: conv+pool — 13×13 → 5×5
        layers.Conv2D(64, (3,3), activation='relu'),  # (11×11×64)
        layers.MaxPooling2D((2,2)),    # (5×5×64)
        # Bloco 3: conv sem pool (retenção de features finais)
        layers.Conv2D(64, (3,3), activation='relu'),  # (3×3×64)
        layers.Flatten(),              # 3×3×64 = 576 elementos → vetor 1D
        layers.Dense(64, activation='relu'),   # classificação de alto nível
        layers.Dropout(0.5),           # desativa 50% dos neurônios — previne overfitting
        layers.Dense(10, activation='softmax') # 10 probabilidades (0-9), somam 1.0
    ])

    # ─── PASSO 4: Compilar ───
    model.compile(
        optimizer='adam',              # Adam: ajuste adaptativo do LR por parâmetro
        loss='categorical_crossentropy',  # métrca para distribuições de probabilidade
        metrics=['accuracy']
    )

    # ─── PASSO 5: Treinar ───
    history = model.fit(
        x_train, y_train,
        epochs=10,           # 10 passagens completas pelo dataset
        batch_size=128,      # atualiza pesos a cada 128 amostras (mini-batch SGD)
        validation_split=0.1 # usa 10% do treino como validação (não entra no treino)
    )

    # ─── PASSO 6: Avaliar no conjunto de TESTE (dados nunca vistos) ───
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
