# GO1201-CnnMnistPassoAPasso
# ═══════════════════════════════════════════════════════════════════
# CNN COMPLETA PARA MNIST - IMPLEMENTAÇÃO PASSO A PASSO
# ═══════════════════════════════════════════════════════════════════

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np

# ─── 1. CARREGAR E EXPLORAR DADOS ───
print("Carregando MNIST...")
# MNIST: 70.000 imagens 28x28 em escala de cinza de dígitos manuscritos (0-9)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(f"Train: {x_train.shape}, Labels: {y_train.shape}")  # (60000, 28, 28)
print(f"Test: {x_test.shape}, Labels: {y_test.shape}")    # (10000, 28, 28)
print(f"Classes: {len(np.unique(y_train))}")  # 10 dígitos (0-9)

# Visualizar exemplos
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_train[i], cmap='gray')
    ax.set_title(f'Label: {y_train[i]}')
    ax.axis('off')
plt.suptitle('MNIST - Exemplos de Treinamento')
plt.tight_layout()
plt.show()

# ─── 2. PRÉ-PROCESSAMENTO ───
# Reshape: CNN espera tensor 4D (amostras, altura, largura, canais) — canal=1 pois MNIST é grayscale
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Normalização: pixels [0,255] → [0,1] para acelerar convergência e evitar explosão de gradientes
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encoding: converte rótulo inteiro em vetor binário (ex: 3 → [0,0,0,1,0,0,0,0,0,0])
# Necessário para usar categorical_crossentropy como função de perda
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print(f"\nApós pré-processamento:")
print(f"x_train: {x_train.shape}, x_test: {x_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

# ─── 3. CONSTRUIR MODELO CNN ───
model = models.Sequential([
    # Bloco 1: Conv → ReLU → MaxPool
    # 32 filtros 3x3: cada filtro aprende um padrão diferente (bordas, curvas, texturas)
    # relu: f(x)=max(0,x) — introduz não-linearidade e evita vanishing gradient da sigmoid
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv1'),
    # MaxPooling 2x2: reduz dimensão pela metade (26x26 → 13x13), preserva feature mais forte
    layers.MaxPooling2D((2, 2), name='pool1'),
    # Output: (13, 13, 32)

    # Bloco 2: mais filtros (64) para capturar padrões mais complexos que o Bloco 1
    layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
    layers.MaxPooling2D((2, 2), name='pool2'),
    # Output: (5, 5, 64)

    # Bloco 3: Conv → ReLU (sem pooling para não reduzir demais a dimensão espacial)
    layers.Conv2D(64, (3, 3), activation='relu', name='conv3'),
    # Output: (3, 3, 64)

    # Flatten: converte tensor 3D (3,3,64) em vetor 1D (576) para entrada nas camadas densas
    layers.Flatten(name='flatten'),

    # Fully Connected: aprende combinações de alto nível das features extraídas pelas Conv
    layers.Dense(64, activation='relu', name='fc1'),
    # Dropout 50%: desativa aleatoriamente metade dos neurônios no treino — previne overfitting
    layers.Dropout(0.5, name='dropout'),

    # Camada de saída: 10 neurônios (um por dígito), softmax converte em probabilidades somando 1
    layers.Dense(10, activation='softmax', name='output')
], name='MNIST_CNN')

# Visualizar arquitetura
model.summary()

# Calcular parâmetros
print("\nNúmero de parâmetros:")
print(f"Total: {model.count_params():,}")
print(f"Treináveis: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

# ─── 4. COMPILAR ───
model.compile(
    # Adam: otimizador adaptativo que ajusta a taxa de aprendizado por parâmetro — robusto e eficiente
    optimizer='adam',
    # categorical_crossentropy: mede divergência entre distribuição predita e real (one-hot)
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ─── 5. CALLBACKS ───
callbacks = [
    # Early stopping: parar se val_loss não melhorar por 3 épocas
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),

    # Model checkpoint: salvar melhor modelo
    keras.callbacks.ModelCheckpoint(
        'best_mnist_cnn.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),

    # Reduce LR: reduzir learning rate se estagnar
    # ReduceLROnPlateau: se val_loss estagnar por 2 épocas, reduz LR pela metade (fator=0.5)
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,   # novo_LR = LR * 0.5
        patience=2,
        verbose=1,
        min_lr=1e-7   # limite inferior para não zerar o aprendizado
    )
]

# ─── 6. TREINAR ───
print("\nTreinando modelo...")
history = model.fit(
    x_train, y_train,
    batch_size=128,       # mini-batch: atualiza pesos a cada 128 amostras (equilíbrio velocidade/precisão)
    epochs=15,            # máximo de passagens completas pelo dataset (EarlyStopping pode interromper antes)
    validation_split=0.1, # reserva 10% do treino como validação para monitorar overfitting
    callbacks=callbacks,
    verbose=1
)

# ─── 7. AVALIAR ───
print("\nAvaliando no conjunto de teste...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")  # ~99.2%

# ─── 8. VISUALIZAR TREINAMENTO ───
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss
ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
ax1.set_title('Model Loss', fontsize=14)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy
ax2.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
ax2.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
ax2.set_title('Model Accuracy', fontsize=14)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ─── 9. PREDIÇÕES E ANÁLISE DE ERROS ───
# Fazer predições: retorna matriz de probabilidades (10000, 10)
y_pred = model.predict(x_test)
# argmax: converte vetor de probabilidades no índice da classe mais provável (ex: [0.01,...,0.98] → 9)
y_pred_classes = np.argmax(y_pred, axis=1)
# Converte one-hot de volta para inteiros para comparação e métricas
y_true_classes = np.argmax(y_test, axis=1)

# Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

cm = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True)
plt.title('Confusion Matrix - MNIST CNN')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, 
                          target_names=[str(i) for i in range(10)]))

# ─── 10. VISUALIZAR PREDIÇÕES CORRETAS E ERRADAS ───
# Encontrar erros
# np.where: encontra índices onde a predição difere do rótulo real
errors = np.where(y_pred_classes != y_true_classes)[0]

print(f"\nTotal de erros: {len(errors)} / {len(y_test)} = {len(errors)/len(y_test)*100:.2f}%")

# Visualizar primeiros 10 erros
if len(errors) > 0:
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for i, ax in enumerate(axes.flat):
        if i < len(errors):
            idx = errors[i]
            ax.imshow(x_test[idx].reshape(28, 28), cmap='gray')
            ax.set_title(f'True: {y_true_classes[idx]}\nPred: {y_pred_classes[idx]}', 
                        color='red', fontsize=12)
            ax.axis('off')
    plt.suptitle('Primeiros 10 Erros de Classificação', fontsize=16)
    plt.tight_layout()
    plt.show()

# Visualizar predições corretas
corrects = np.where(y_pred_classes == y_true_classes)[0]
sample_corrects = np.random.choice(corrects, 10, replace=False)

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for i, ax in enumerate(axes.flat):
    idx = sample_corrects[i]
    ax.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    ax.set_title(f'Pred: {y_pred_classes[idx]} ✓', color='green', fontsize=12)
    ax.axis('off')
plt.suptitle('Predições Corretas', fontsize=16)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════
# RESULTADO ESPERADO:
# • Test Accuracy: ~99.2%
# • Principais erros: dígitos confusos (3 vs 5, 4 vs 9, 7 vs 1)
# • Tempo de treino: ~5 min em CPU, <1 min em GPU
# ═══════════════════════════════════════════════════════════════════
