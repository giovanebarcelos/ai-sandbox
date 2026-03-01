# GO1203-11cTransferLearningResnet50
# ═══════════════════════════════════════════════════════════════════
# TRANSFER LEARNING COM RESNET50 PRÉ-TREINADO
# Classificação de imagens CIFAR-10
# ═══════════════════════════════════════════════════════════════════

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
import numpy as np

# ─── 1. CARREGAR CIFAR-10 ───
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Classes do CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Train: {x_train.shape}, Test: {x_test.shape}")

# ─── 2. PRÉ-PROCESSAMENTO ───
# ResNet50 espera input 224x224 (vamos usar 32x32 mesmo e fazer resize)
# Normalizar para [-1, 1] (padrão ResNet)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

from tensorflow.keras.applications.resnet50 import preprocess_input
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

# One-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# ─── 3. CARREGAR RESNET50 PRÉ-TREINADO ───
print("\nCarregando ResNet50 pré-treinado (ImageNet)...")

base_model = ResNet50(
    weights='imagenet',      # Pesos do ImageNet
    include_top=False,       # Remover camadas FC originais
    input_shape=(32, 32, 3)  # CIFAR-10 é 32x32
)

# Congelar todas as camadas do ResNet (Feature Extraction)
base_model.trainable = False

print(f"Base model: {len(base_model.layers)} camadas congeladas")

# ─── 4. ADICIONAR CLASSIFICADOR CUSTOMIZADO ───
model_transfer = models.Sequential([
    base_model,

    # Global Average Pooling (reduz dimensão)
    layers.GlobalAveragePooling2D(),

    # Camadas densas
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
], name='ResNet50_CIFAR10')

model_transfer.summary()

# ─── 5. COMPILAR E TREINAR (FASE 1: Feature Extraction) ───
model_transfer.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🔵 FASE 1: Feature Extraction (base congelada)")
history_phase1 = model_transfer.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_data=(x_test, y_test),
    verbose=1
)

test_loss_p1, test_acc_p1 = model_transfer.evaluate(x_test, y_test, verbose=0)
print(f"Fase 1 - Test Accuracy: {test_acc_p1:.4f}")

# ─── 6. FINE-TUNING (FASE 2: Descongelar últimas camadas) ───
print("\n🔴 FASE 2: Fine-Tuning (descongelar últimas 10 camadas)")

# Descongelar últimas camadas
base_model.trainable = True

# Congelar todas exceto últimas 10
for layer in base_model.layers[:-10]:
    layer.trainable = False

print(f"Camadas treináveis: {len([l for l in base_model.layers if l.trainable])}")

# Recompilar com learning rate menor
model_transfer.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # LR pequeno!
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_phase2 = model_transfer.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_data=(x_test, y_test),
    verbose=1
)

test_loss_p2, test_acc_p2 = model_transfer.evaluate(x_test, y_test, verbose=0)
print(f"Fase 2 - Test Accuracy: {test_acc_p2:.4f}")

# ─── 7. COMPARAÇÃO ───
print("\n" + "="*60)
print("RESULTADOS - TRANSFER LEARNING")
print("="*60)
print(f"Fase 1 (Feature Extraction): {test_acc_p1:.4f}")
print(f"Fase 2 (Fine-Tuning):        {test_acc_p2:.4f}")
print(f"Melhoria: {(test_acc_p2 - test_acc_p1)*100:.2f}%")
print("="*60)

# ─── 8. VISUALIZAR CURVAS ───
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history_phase1.history['loss'], label='Phase 1 Train')
plt.plot(history_phase1.history['val_loss'], label='Phase 1 Val')
plt.plot(range(10, 20), history_phase2.history['loss'], label='Phase 2 Train')
plt.plot(range(10, 20), history_phase2.history['val_loss'], label='Phase 2 Val')
plt.axvline(x=10, color='red', linestyle='--', label='Fine-tuning starts')
plt.title('Loss - Transfer Learning')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history_phase1.history['accuracy'], label='Phase 1 Train')
plt.plot(history_phase1.history['val_accuracy'], label='Phase 1 Val')
plt.plot(range(10, 20), history_phase2.history['accuracy'], label='Phase 2 Train')
plt.plot(range(10, 20), history_phase2.history['val_accuracy'], label='Phase 2 Val')
plt.axvline(x=10, color='red', linestyle='--', label='Fine-tuning starts')
plt.title('Accuracy - Transfer Learning')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════
# OBSERVAÇÕES:
# • Transfer Learning economiza MUITO tempo e dados
# • Fase 1: treina rápido, accuracy razoável
# • Fase 2 (fine-tuning): melhora significativamente
# • ResNet50 pré-treinado já conhece features gerais (bordas, texturas)
# • Adaptamos para CIFAR-10 com apenas camadas finais
# ═══════════════════════════════════════════════════════════════════
