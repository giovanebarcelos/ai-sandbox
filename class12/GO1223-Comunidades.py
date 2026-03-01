# GO1223-Comunidades
# ═══════════════════════════════════════════════════════════════════
# TRANSFER LEARNING E FINE-TUNING ESTRATÉGICO COM RESNET50
# Técnicas avançadas para maximizar performance com poucos dados
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

print("🚀 TRANSFER LEARNING E FINE-TUNING ESTRATÉGICO")
print("=" * 70)

# ─── 1. CONFIGURAÇÕES ───
IMG_SIZE = (224, 224)
NUM_CLASSES = 5
EPOCHS_FROZEN = 10
EPOCHS_FINETUNE = 20

print(f"\n📊 Configurações:")
print(f"  Classes: {NUM_CLASSES}")
print(f"  Epochs frozen: {EPOCHS_FROZEN}")
print(f"  Epochs fine-tune: {EPOCHS_FINETUNE}")

# ─── 2. CARREGAR MODELO BASE PRÉ-TREINADO ───
print("\n🔨 Carregando ResNet50 pré-treinado (ImageNet)...")

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)

print(f"  Total de camadas: {len(base_model.layers)}")
print(f"  Parâmetros totais: {base_model.count_params():,}")

# ─── 3. CONSTRUIR CLASSIFICADOR CUSTOMIZADO ───
base_model.trainable = False  # Congelar base inicialmente

x = base_model.output
x = GlobalAveragePooling2D(name='global_avg_pool')(x)
x = BatchNormalization(name='bn1')(x)
x = Dense(512, activation='relu', name='dense_512')(x)
x = Dropout(0.5, name='dropout1')(x)
x = BatchNormalization(name='bn2')(x)
x = Dense(256, activation='relu', name='dense_256')(x)
x = Dropout(0.3, name='dropout2')(x)
outputs = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

model = Model(inputs=base_model.input, outputs=outputs, name='resnet50_transfer')

print(f"\n📋 Arquitetura:")
print(f"  Base: ResNet50 (frozen)")
print(f"  Top: GAP → BN → Dense(512) → Dropout → Dense(256) → Dense({NUM_CLASSES})")
print(f"  Total parâmetros: {model.count_params():,}")

# ─── 4. COMPILAR PARA FASE 1 (BASE FROZEN) ───
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ─── 5. DADOS DUMMY PARA DEMONSTRAÇÃO ───
X_train = np.random.rand(100, *IMG_SIZE, 3)
y_train = tf.keras.utils.to_categorical(np.random.randint(0, NUM_CLASSES, 100), NUM_CLASSES)
X_val = np.random.rand(30, *IMG_SIZE, 3)
y_val = tf.keras.utils.to_categorical(np.random.randint(0, NUM_CLASSES, 30), NUM_CLASSES)

# ─── 6. FASE 1: TREINAR APENAS TOP LAYERS ───
print("\n" + "="*70)
print("📚 FASE 1: TREINANDO APENAS CAMADAS SUPERIORES (base frozen)")
print("="*70)

callbacks_phase1 = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ModelCheckpoint('resnet50_phase1.h5', monitor='val_accuracy', save_best_only=True)
]

history_phase1 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,  # Reduzido para demonstração
    batch_size=32,
    callbacks=callbacks_phase1,
    verbose=1
)

print(f"\n✅ Fase 1 completa! Val Accuracy: {max(history_phase1.history['val_accuracy']):.4f}")

# ─── 7. FASE 2: FINE-TUNING DE CAMADAS PROFUNDAS ───
print("\n" + "="*70)
print("🔧 FASE 2: FINE-TUNING (descongelar últimas camadas)")
print("="*70)

base_model.trainable = True

# Congelar tudo exceto últimas 30 layers
freeze_until = len(base_model.layers) - 30
for layer in base_model.layers[:freeze_until]:
    layer.trainable = False

print(f"\n📊 Estratégia:")
print(f"  Total layers: {len(base_model.layers)}")
print(f"  Frozen: {freeze_until}")
print(f"  Trainable: {len(base_model.layers) - freeze_until}")

# Recompilar com LR muito menor
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # 100x menor!
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_phase2 = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-8, verbose=1),
    ModelCheckpoint('resnet50_finetuned.h5', monitor='val_accuracy', save_best_only=True)
]

history_phase2 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=32,
    callbacks=callbacks_phase2,
    verbose=1
)

print(f"\n✅ Fine-tuning completo! Val Accuracy: {max(history_phase2.history['val_accuracy']):.4f}")

# ─── 8. VISUALIZAR COMPARAÇÃO ───
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Accuracy - Fase 1
axes[0, 0].plot(history_phase1.history['accuracy'], label='Train', linewidth=2)
axes[0, 0].plot(history_phase1.history['val_accuracy'], label='Val', linewidth=2)
axes[0, 0].set_title('Fase 1: Accuracy (Base Frozen)', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss - Fase 1
axes[0, 1].plot(history_phase1.history['loss'], label='Train', linewidth=2)
axes[0, 1].plot(history_phase1.history['val_loss'], label='Val', linewidth=2)
axes[0, 1].set_title('Fase 1: Loss (Base Frozen)', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Accuracy - Fase 2
axes[1, 0].plot(history_phase2.history['accuracy'], label='Train', linewidth=2)
axes[1, 0].plot(history_phase2.history['val_accuracy'], label='Val', linewidth=2)
axes[1, 0].set_title('Fase 2: Accuracy (Fine-tuned)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Loss - Fase 2
axes[1, 1].plot(history_phase2.history['loss'], label='Train', linewidth=2)
axes[1, 1].plot(history_phase2.history['val_loss'], label='Val', linewidth=2)
axes[1, 1].set_title('Fase 2: Loss (Fine-tuned)', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('transfer_learning_comparison.png', dpi=150)
print("✅ Comparação salva: transfer_learning_comparison.png")

# ─── 9. ANÁLISE FINAL ───
phase1_best = max(history_phase1.history['val_accuracy'])
phase2_best = max(history_phase2.history['val_accuracy'])
improvement = (phase2_best - phase1_best) / phase1_best * 100

print("\n" + "="*70)
print("📊 ANÁLISE FINAL")
print("="*70)
print(f"\n🎯 Resultados:")
print(f"  Fase 1 (Frozen): {phase1_best:.4f}")
print(f"  Fase 2 (Fine-tuned): {phase2_best:.4f}")
print(f"  Melhoria: +{improvement:.2f}%")

print(f"\n💡 Estratégia aplicada:")
print(f"  1. Base frozen → treinar apenas top layers (rápido)")
print(f"  2. Descongelar últimas 30 layers → fine-tuning (preciso)")
print(f"  3. LR 100x menor na fase 2 → evitar catastrophic forgetting")

print(f"\n📚 Quando usar Transfer Learning:")
print(f"  ✓ Dataset pequeno (<10k images)")
print(f"  ✓ Domínio similar a ImageNet")
print(f"  ✓ Recursos computacionais limitados")
print(f"  ✓ Tempo de treinamento curto")

print(f"\n⚡ Ganhos típicos:")
print(f"  • Redução de tempo: 10-50x")
print(f"  • Accuracy: +5-15% vs treinar do zero")
print(f"  • Dados necessários: 10-100x menos")

print("\n✅ TRANSFER LEARNING COMPLETO!")
