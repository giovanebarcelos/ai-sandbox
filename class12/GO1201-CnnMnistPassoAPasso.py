# GO1201-CnnMnistPassoAPasso
# ═══════════════════════════════════════════════════════════════════
# CNN COMPLETA PARA MNIST - IMPLEMENTAÇÃO PASSO A PASSO
# ═══════════════════════════════════════════════════════════════════

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# ─── 1. CARREGAR E EXPLORAR DADOS ───
print("Carregando MNIST...")
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
# Reshape: adicionar canal (28, 28) → (28, 28, 1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Normalizar: [0, 255] → [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encoding
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print(f"\nApós pré-processamento:")
print(f"x_train: {x_train.shape}, x_test: {x_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

# ─── 3. CONSTRUIR MODELO CNN ───
model = models.Sequential([
    # Bloco 1: Conv → ReLU → MaxPool
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv1'),
    layers.MaxPooling2D((2, 2), name='pool1'),
    # Output: (13, 13, 32)

    # Bloco 2: Conv → ReLU → MaxPool
    layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
    layers.MaxPooling2D((2, 2), name='pool2'),
    # Output: (5, 5, 64)

    # Bloco 3: Conv → ReLU
    layers.Conv2D(64, (3, 3), activation='relu', name='conv3'),
    # Output: (3, 3, 64)

    # Flatten: (3, 3, 64) → (576)
    layers.Flatten(name='flatten'),

    # Fully Connected
    layers.Dense(64, activation='relu', name='fc1'),
    layers.Dropout(0.5, name='dropout'),

    # Output layer
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
    optimizer='adam',
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
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        verbose=1,
        min_lr=1e-7
    )
]

# ─── 6. TREINAR ───
print("\nTreinando modelo...")
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=15,
    validation_split=0.1,
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
# Fazer predições
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

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
