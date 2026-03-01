# GO1211-35iMixedPrecisionTrainingAceleraçãoCom
# ═══════════════════════════════════════════════════════════════════
# MIXED PRECISION TRAINING - TREINO RÁPIDO COM FP16
# Usar float16 para acelerar treino mantendo precisão
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt
import time

print("⚡ MIXED PRECISION TRAINING")
print("=" * 70)

# ─── 1. VERIFICAR GPU ───
print("\n🔍 Verificando GPU...")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"  ✓ GPU disponível: {len(gpus)} device(s)")
    for gpu in gpus:
        print(f"    {gpu}")
else:
    print("  ⚠️  Nenhuma GPU detectada (rodará em CPU)")

# ─── 2. CONFIGURAR MIXED PRECISION ───
print("\n⚙️  Configurando Mixed Precision...")

# Ativar mixed precision policy
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

print(f"  Compute dtype: {policy.compute_dtype}")
print(f"  Variable dtype: {policy.variable_dtype}")
print("  ✓ Mixed precision ativado (float16 + float32)")

# ─── 3. CARREGAR DADOS ───
print("\n📦 Carregando CIFAR-10...")

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalizar
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reduzir para comparação rápida
X_train = X_train[:10000]
y_train = y_train[:10000]
X_test = X_test[:2000]
y_test = y_test[:2000]

print(f"  Train: {X_train.shape}")
print(f"  Test: {X_test.shape}")

# ─── 4. CRIAR MODELO ───
print("\n🏗️ Construindo modelo CNN...")

def create_model():
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax', dtype='float32')  # Output em float32
    ], name='CNN_CIFAR10')

    return model

model_fp16 = create_model()

print(f"  Parâmetros: {model_fp16.count_params():,}")

# Compilar com loss scaling (importante para FP16)
optimizer = tf.keras.optimizers.Adam()
optimizer = mixed_precision.LossScaleOptimizer(optimizer)

model_fp16.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("  ✓ Modelo compilado com mixed precision")

# ─── 5. TREINAR COM FP16 ───
print("\n🚀 Treinando com Mixed Precision (FP16)...")

start_time = time.time()

history_fp16 = model_fp16.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=128,
    verbose=0
)

time_fp16 = time.time() - start_time

test_loss_fp16, test_acc_fp16 = model_fp16.evaluate(X_test, y_test, verbose=0)

print(f"  Tempo de treino: {time_fp16:.2f}s")
print(f"  Test Accuracy: {test_acc_fp16:.4f}")
print(f"  Final Loss: {history_fp16.history['loss'][-1]:.4f}")

# ─── 6. COMPARAR COM FP32 (BASELINE) ───
print("\n🐢 Treinando com Float32 (baseline)...")

# Desativar mixed precision
mixed_precision.set_global_policy('float32')

model_fp32 = create_model()
model_fp32.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

start_time = time.time()

history_fp32 = model_fp32.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=128,
    verbose=0
)

time_fp32 = time.time() - start_time

test_loss_fp32, test_acc_fp32 = model_fp32.evaluate(X_test, y_test, verbose=0)

print(f"  Tempo de treino: {time_fp32:.2f}s")
print(f"  Test Accuracy: {test_acc_fp32:.4f}")
print(f"  Final Loss: {history_fp32.history['loss'][-1]:.4f}")

# ─── 7. ANÁLISE DE PERFORMANCE ───
print("\n" + "="*70)
print("📊 COMPARAÇÃO: FP16 vs FP32")
print("="*70)

speedup = time_fp32 / time_fp16
acc_diff = abs(test_acc_fp16 - test_acc_fp32)

print(f"\n⏱️  TEMPO:")
print(f"  FP32 (baseline): {time_fp32:.2f}s")
print(f"  FP16 (mixed):    {time_fp16:.2f}s")
print(f"  Speedup:         {speedup:.2f}x mais rápido")

print(f"\n🎯 ACCURACY:")
print(f"  FP32: {test_acc_fp32:.4f}")
print(f"  FP16: {test_acc_fp16:.4f}")
print(f"  Diferença: {acc_diff:.4f} ({acc_diff*100:.2f}%)")

print(f"\n💾 MEMÓRIA (estimado):")
print(f"  FP32: ~100% (baseline)")
print(f"  FP16: ~50% (metade do tamanho)")

# ─── 8. VISUALIZAR ───
print("\n📈 Visualizando comparação...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Training loss
axes[0, 0].plot(history_fp32.history['loss'], label='FP32', linewidth=2)
axes[0, 0].plot(history_fp16.history['loss'], label='FP16', linewidth=2, linestyle='--')
axes[0, 0].set_title('Training Loss', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Training accuracy
axes[0, 1].plot(history_fp32.history['accuracy'], label='FP32', linewidth=2)
axes[0, 1].plot(history_fp16.history['accuracy'], label='FP16', linewidth=2, linestyle='--')
axes[0, 1].set_title('Training Accuracy', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Tempo comparação
axes[1, 0].bar(['FP32', 'FP16'], [time_fp32, time_fp16], color=['steelblue', 'orange'], alpha=0.7)
axes[1, 0].set_title(f'Tempo de Treino (Speedup: {speedup:.2f}x)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Tempo (s)')
axes[1, 0].grid(axis='y', alpha=0.3)

# Accuracy comparação
axes[1, 1].bar(['FP32', 'FP16'], [test_acc_fp32, test_acc_fp16], color=['steelblue', 'orange'], alpha=0.7)
axes[1, 1].set_title('Test Accuracy', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_ylim(0, 1)
axes[1, 1].grid(axis='y', alpha=0.3)

plt.suptitle('Mixed Precision Training (FP16 vs FP32)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('mixed_precision_comparison.png', dpi=150)
print("✅ Comparação salva: mixed_precision_comparison.png")

print("\n💡 COMO FUNCIONA:")
print("  • Compute (forward/backward): Float16 (FP16) - mais rápido")
print("  • Variables (weights): Float32 (FP32) - mais preciso")
print("  • Loss Scaling: Evita underflow em gradientes pequenos")
print("  • Output Layer: Sempre FP32 (estabilidade numérica)")

print("\n📚 QUANDO USAR:")
print("  ✓ GPUs modernas (Tensor Cores: V100, A100, RTX)")
print("  ✓ Modelos grandes (ResNet, Transformer)")
print("  ✓ Datasets grandes (treino longo)")
print("  ✗ Modelos muito pequenos (overhead não compensa)")
print("  ✗ CPU (não suportado)")

print("\n🎯 BENEFÍCIOS:")
print("  • 1.5-3x mais rápido (com GPU compatível)")
print("  • 50% menos memória (batch maior ou modelo maior)")
print("  • Accuracy praticamente idêntica (<0.1% diferença)")
print("  • Menor custo de treino em cloud")

print("\n✅ MIXED PRECISION COMPLETO!")
