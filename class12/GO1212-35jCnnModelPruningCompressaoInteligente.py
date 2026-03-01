# GO1212-35jCnnModelPruningCompressãoInteligente
# ═══════════════════════════════════════════════════════════════════
# CNN MODEL PRUNING - COMPRESSÃO INTELIGENTE
# Remover pesos desnecessários para modelos menores e rápidos
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
import tensorflow_model_optimization as tfmot
import matplotlib.pyplot as plt
import time

print("✂️ CNN MODEL PRUNING")
print("=" * 70)

# ─── 1. CARREGAR DADOS ───
print("\n📦 Carregando MNIST...")

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Reduzir para treino rápido
X_train = X_train[:10000]
y_train = y_train[:10000]

print(f"  Train: {X_train.shape}")
print(f"  Test: {X_test.shape}")

# ─── 2. TREINAR MODELO BASE ───
print("\n🏗️ Treinando modelo base (denso)...")

base_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
], name='BaseModel')

base_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(f"  Parâmetros: {base_model.count_params():,}")

history_base = base_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=128,
    verbose=0
)

base_acc = base_model.evaluate(X_test, y_test, verbose=0)[1]
print(f"  Base Accuracy: {base_acc:.4f}")

# ─── 3. MAGNITUDE-BASED PRUNING ───
print("\n✂️ Aplicando Magnitude-Based Pruning...")

# Configurar pruning (50% sparsity)
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,  # 50% dos pesos zerados
        begin_step=0,
        end_step=1000
    )
}

model_for_pruning = prune_low_magnitude(base_model, **pruning_params)

model_for_pruning.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"  Target sparsity: 50%")
print(f"  Parâmetros (com masks): {model_for_pruning.count_params():,}")

# ─── 4. TREINAR COM PRUNING ───
print("\n🚀 Fine-tuning com pruning...")

callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep()
]

history_pruning = model_for_pruning.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=3,
    batch_size=128,
    callbacks=callbacks,
    verbose=0
)

pruned_acc = model_for_pruning.evaluate(X_test, y_test, verbose=0)[1]
print(f"  Pruned Accuracy: {pruned_acc:.4f}")

# ─── 5. EXPORTAR MODELO PRUNED ───
print("\n💾 Exportando modelo pruned...")

model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

print(f"  Parâmetros (sem masks): {model_for_export.count_params():,}")
print("  ✓ Modelo pruned exportado")

# ─── 6. ANÁLISE DE SPARSITY ───
print("\n🔍 Analisando sparsity por camada...")

for layer in model_for_export.layers:
    if hasattr(layer, 'kernel'):
        weights = layer.get_weights()[0]
        total = weights.size
        zeros = np.sum(weights == 0)
        sparsity = zeros / total

        print(f"  {layer.name:20s}: {sparsity:.2%} sparse ({zeros}/{total} zeros)")

# ─── 7. COMPARAÇÃO DE INFERÊNCIA ───
print("\n⚡ Comparando velocidade de inferência...")

# Benchmark
test_batch = X_test[:100]

# Base model
start = time.time()
for _ in range(10):
    _ = base_model.predict(test_batch, verbose=0)
time_base = (time.time() - start) / 10

# Pruned model
start = time.time()
for _ in range(10):
    _ = model_for_export.predict(test_batch, verbose=0)
time_pruned = (time.time() - start) / 10

speedup = time_base / time_pruned

print(f"  Base model:   {time_base*1000:.2f}ms")
print(f"  Pruned model: {time_pruned*1000:.2f}ms")
print(f"  Speedup:      {speedup:.2f}x")

# ─── 8. VISUALIZAR ───
print("\n📊 Visualizando pesos...")

# Pegar pesos da primeira camada Conv
base_weights = base_model.layers[0].get_weights()[0]
pruned_weights = model_for_export.layers[0].get_weights()[0]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Base model filters
for i in range(4):
    filter_base = base_weights[:, :, 0, i]
    axes[0, i].imshow(filter_base, cmap='viridis')
    axes[0, i].set_title(f'Base Filter {i}', fontsize=10, fontweight='bold')
    axes[0, i].axis('off')

# Pruned model filters
for i in range(4):
    filter_pruned = pruned_weights[:, :, 0, i]
    axes[1, i].imshow(filter_pruned, cmap='viridis')

    zeros = np.sum(filter_pruned == 0)
    total = filter_pruned.size
    sparsity = zeros / total

    axes[1, i].set_title(f'Pruned Filter {i}\n({sparsity:.1%} sparse)', fontsize=10, fontweight='bold')
    axes[1, i].axis('off')

plt.suptitle('Weight Pruning Visualization', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('model_pruning_weights.png', dpi=150)
print("✅ Pesos salvos: model_pruning_weights.png")

# Comparação de métricas
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Accuracy
axes[0].bar(['Base', 'Pruned (50%)'], [base_acc, pruned_acc], color=['steelblue', 'orange'], alpha=0.7)
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
axes[0].set_ylim(0.9, 1.0)
axes[0].grid(axis='y', alpha=0.3)

# Parâmetros efetivos
base_params = base_model.count_params()
pruned_effective = base_params * 0.5  # 50% sparse

axes[1].bar(['Base', 'Pruned (50%)'], [base_params, pruned_effective], color=['steelblue', 'orange'], alpha=0.7)
axes[1].set_ylabel('Parâmetros Efetivos')
axes[1].set_title('Model Size', fontsize=12, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

# Tempo de inferência
axes[2].bar(['Base', 'Pruned'], [time_base*1000, time_pruned*1000], color=['steelblue', 'orange'], alpha=0.7)
axes[2].set_ylabel('Tempo (ms)')
axes[2].set_title(f'Inference Time (Speedup: {speedup:.2f}x)', fontsize=12, fontweight='bold')
axes[2].grid(axis='y', alpha=0.3)

plt.suptitle('Model Pruning - Performance Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('model_pruning_comparison.png', dpi=150)
print("✅ Comparação salva: model_pruning_comparison.png")

# ─── 9. RESUMO ───
print("\n" + "="*70)
print("📊 RESUMO DO PRUNING")
print("="*70)

print(f"\n🎯 ACCURACY:")
print(f"  Base:   {base_acc:.4f}")
print(f"  Pruned: {pruned_acc:.4f}")
print(f"  Perda:  {(base_acc - pruned_acc):.4f} ({(base_acc - pruned_acc)*100:.2f}%)")

print(f"\n💾 TAMANHO:")
print(f"  Base:   {base_params:,} parâmetros (100%)")
print(f"  Pruned: {int(pruned_effective):,} parâmetros efetivos (50%)")
print(f"  Redução: 50% menor")

print(f"\n⚡ PERFORMANCE:")
print(f"  Speedup: {speedup:.2f}x mais rápido")
print(f"  Latência: {time_pruned*1000:.2f}ms vs {time_base*1000:.2f}ms")

print("\n💡 TÉCNICAS DE PRUNING:")
print("  • Magnitude-Based: Remove pesos pequenos (mais simples)")
print("  • Structured: Remove filtros/canais inteiros")
print("  • Iterative: Pruning + fine-tuning repetido")
print("  • Dynamic: Pruning durante treino (Lottery Ticket)")

print("\n📚 QUANDO USAR:")
print("  ✓ Deploy em dispositivos edge (mobile, IoT)")
print("  ✓ Reduzir latência de inferência")
print("  ✓ Economizar memória e bandwidth")
print("  ✓ Modelos over-parameterizados")

print("\n🎯 SPARSITY TÍPICA:")
print("  • 50-70%: Mínima perda de accuracy")
print("  • 80-90%: Perda moderada (1-3%)")
print("  • 95%+: Perda significativa ou requer técnicas avançadas")

print("\n✅ MODEL PRUNING COMPLETO!")
