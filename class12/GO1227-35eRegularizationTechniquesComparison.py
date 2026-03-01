# GO1227-35eRegularizationTechniquesComparison
# ═══════════════════════════════════════════════════════════════════
# TÉCNICAS DE REGULARIZAÇÃO EM CNNs - COMPARAÇÃO PRÁTICA
# Dropout, Batch Normalization, L2 Regularization, Early Stopping
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

print("🛡️ REGULARIZATION TECHNIQUES COMPARISON")
print("=" * 70)

# ─── 1. CONFIGURAÇÕES ───
INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10
EPOCHS = 20
BATCH_SIZE = 32

print(f"\n📊 Configurações:")
print(f"  Input shape: {INPUT_SHAPE}")
print(f"  Classes: {NUM_CLASSES}")
print(f"  Epochs: {EPOCHS}")

# ─── 2. CRIAR DADOS SINTÉTICOS ───
print("\n📦 Criando dados de treinamento...")

X_train = np.random.rand(1000, *INPUT_SHAPE)
y_train = np.random.randint(0, NUM_CLASSES, 1000)
X_val = np.random.rand(200, *INPUT_SHAPE)
y_val = np.random.randint(0, NUM_CLASSES, 200)

print(f"  Train: {X_train.shape}")
print(f"  Val: {X_val.shape}")

# ─── 3. MODELO BASE (SEM REGULARIZAÇÃO) ───
print("\n🔧 Modelo 1: BASE (sem regularização)")

model_base = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
], name='Base')

model_base.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_base = model_base.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0
)

print(f"  Parâmetros: {model_base.count_params():,}")
print(f"  Val Accuracy: {max(history_base.history['val_accuracy']):.4f}")

# ─── 4. MODELO COM DROPOUT ───
print("\n🔧 Modelo 2: DROPOUT (0.5)")

model_dropout = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
], name='Dropout')

model_dropout.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_dropout = model_dropout.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0
)

print(f"  Parâmetros: {model_dropout.count_params():,}")
print(f"  Val Accuracy: {max(history_dropout.history['val_accuracy']):.4f}")

# ─── 5. MODELO COM BATCH NORMALIZATION ───
print("\n🔧 Modelo 3: BATCH NORMALIZATION")

model_bn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    Flatten(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(NUM_CLASSES, activation='softmax')
], name='BatchNorm')

model_bn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_bn = model_bn.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0
)

print(f"  Parâmetros: {model_bn.count_params():,}")
print(f"  Val Accuracy: {max(history_bn.history['val_accuracy']):.4f}")

# ─── 6. MODELO COM L2 REGULARIZATION ───
print("\n🔧 Modelo 4: L2 REGULARIZATION (0.01)")

model_l2 = Sequential([
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01), input_shape=INPUT_SHAPE),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    Flatten(),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(NUM_CLASSES, activation='softmax')
], name='L2_Reg')

model_l2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_l2 = model_l2.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0
)

print(f"  Parâmetros: {model_l2.count_params():,}")
print(f"  Val Accuracy: {max(history_l2.history['val_accuracy']):.4f}")

# ─── 7. MODELO COM TODAS AS TÉCNICAS ───
print("\n🔧 Modelo 5: COMBO (Dropout + BN + L2)")

model_combo = Sequential([
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001), input_shape=INPUT_SHAPE),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Flatten(),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
], name='Combo')

model_combo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history_combo = model_combo.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=0
)

print(f"  Parâmetros: {model_combo.count_params():,}")
print(f"  Val Accuracy: {max(history_combo.history['val_accuracy']):.4f}")
print(f"  Epochs trained: {len(history_combo.history['loss'])}")

# ─── 8. VISUALIZAR COMPARAÇÃO ───
print("\n📊 Gerando comparação visual...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

histories = {
    'Base (No Reg)': history_base,
    'Dropout': history_dropout,
    'Batch Norm': history_bn,
    'L2 Reg': history_l2,
    'Combo': history_combo
}

colors = ['red', 'blue', 'green', 'orange', 'purple']

# Training Loss
for (name, hist), color in zip(histories.items(), colors):
    axes[0, 0].plot(hist.history['loss'], label=name, color=color, linewidth=2)
axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Validation Loss
for (name, hist), color in zip(histories.items(), colors):
    axes[0, 1].plot(hist.history['val_loss'], label=name, color=color, linewidth=2)
axes[0, 1].set_title('Validation Loss', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Training Accuracy
for (name, hist), color in zip(histories.items(), colors):
    axes[1, 0].plot(hist.history['accuracy'], label=name, color=color, linewidth=2)
axes[1, 0].set_title('Training Accuracy', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Validation Accuracy
for (name, hist), color in zip(histories.items(), colors):
    axes[1, 1].plot(hist.history['val_accuracy'], label=name, color=color, linewidth=2)
axes[1, 1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Regularization Techniques - Comparação de Performance', 
            fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('regularization_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Comparação salva: regularization_comparison.png")

# ─── 9. ANÁLISE FINAL ───
print("\n" + "="*70)
print("📊 ANÁLISE DE RESULTADOS")
print("="*70)

results_summary = []
for name, hist in histories.items():
    best_val_acc = max(hist.history['val_accuracy'])
    best_val_loss = min(hist.history['val_loss'])
    final_train_acc = hist.history['accuracy'][-1]
    overfitting = final_train_acc - best_val_acc

    results_summary.append({
        'name': name,
        'val_acc': best_val_acc,
        'val_loss': best_val_loss,
        'overfit': overfitting
    })

print(f"\n{'Modelo':<20} │ {'Val Acc':>8} │ {'Val Loss':>8} │ {'Overfit':>8}")
print("─" * 70)
for r in results_summary:
    print(f"{r['name']:<20} │ {r['val_acc']:>8.4f} │ {r['val_loss']:>8.4f} │ {r['overfit']:>8.4f}")

print("\n💡 CONCLUSÕES:")
print("  🔴 Base: Alto overfitting (diferença train-val grande)")
print("  🔵 Dropout: Reduz overfitting, pode reduzir capacity")
print("  🟢 Batch Norm: Acelera convergência, estabiliza treinamento")
print("  🟠 L2 Reg: Penaliza pesos grandes, suaviza decisões")
print("  🟣 Combo: Melhor balance entre regularização e performance")

print("\n📚 QUANDO USAR CADA TÉCNICA:")
print("  • Dropout: Quando modelo tem muitos parâmetros")
print("  • Batch Norm: Quase sempre (default em CNNs modernas)")
print("  • L2 Reg: Complemento ao dropout, evita pesos extremos")
print("  • Early Stopping: Sempre! Para no momento ideal")
print("  • Combo: Dataset pequeno com alto risco de overfitting")

print("\n✅ REGULARIZATION COMPARISON COMPLETA!")
