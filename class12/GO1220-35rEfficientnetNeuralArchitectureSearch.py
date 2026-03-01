# GO1220-35rEfficientnetNeuralArchitectureSearch
# ══════════════════════════════════════════════════════════════════
# EFFICIENTNET - COMPOUND SCALING
# Escalar largura, profundidade e resolução simultaneamente
# ══════════════════════════════════════════════════════════════════

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import time

print("🔬 EFFICIENTNET - COMPOUND SCALING")
print("=" * 70)

# ─── 1. CARREGAR DADOS ───
print("\n📦 Carregando CIFAR-10...")

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Subset
X_train = X_train[:5000]
y_train = y_train[:5000]
X_test = X_test[:1000]
y_test = y_test[:1000]

print(f"  Train: {X_train.shape}")
print(f"  Test: {X_test.shape}")

# Resize para EfficientNet (mínimo 32x32, mas funciona melhor com 224x224)
from tensorflow.keras.preprocessing.image import smart_resize

X_train_resized = tf.image.resize(X_train, (96, 96)).numpy()
X_test_resized = tf.image.resize(X_test, (96, 96)).numpy()

print(f"  Resized to: {X_train_resized.shape[1:3]}")

# ─── 2. BASELINE CNN ───
print("\n📊 Baseline: CNN simples...")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

baseline = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
], name='Baseline')

baseline.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

start = time.time()
baseline.fit(X_train, y_train, epochs=10, batch_size=64, verbose=0)
baseline_time = time.time() - start

baseline_acc = baseline.evaluate(X_test, y_test, verbose=0)[1]
baseline_params = baseline.count_params()

print(f"  Accuracy: {baseline_acc:.4f}")
print(f"  Params: {baseline_params:,}")
print(f"  Training time: {baseline_time:.1f}s")

# ─── 3. EFFICIENTNET-B0 ───
print("\n🚀 EfficientNet-B0...")

# Carregar EfficientNet pré-treinado
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(96, 96, 3))

# Congelar base
base_model.trainable = False

# Adicionar cabeçalho
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output = Dense(10, activation='softmax')(x)

efficientnet = Model(inputs=base_model.input, outputs=output, name='EfficientNet-B0')

efficientnet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

start = time.time()
efficientnet.fit(X_train_resized, y_train, epochs=10, batch_size=64, verbose=0)
efficientnet_time = time.time() - start

efficientnet_acc = efficientnet.evaluate(X_test_resized, y_test, verbose=0)[1]
efficientnet_params = efficientnet.count_params()

print(f"  Accuracy: {efficientnet_acc:.4f}")
print(f"  Params: {efficientnet_params:,}")
print(f"  Training time: {efficientnet_time:.1f}s")

# ─── 4. COMPARAR ───
print("\n📊 Comparação...")

models_comparison = {
    'Baseline CNN': {'acc': baseline_acc, 'params': baseline_params, 'time': baseline_time},
    'EfficientNet-B0': {'acc': efficientnet_acc, 'params': efficientnet_params, 'time': efficientnet_time}
}

for name, metrics in models_comparison.items():
    print(f"\n  {name}:")
    print(f"    Accuracy: {metrics['acc']:.4f}")
    print(f"    Params: {metrics['params']:,}")
    print(f"    Time: {metrics['time']:.1f}s")

# ─── 5. VISUALIZAR ───
print("\n📈 Visualizando comparação...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

models = list(models_comparison.keys())
accs = [m['acc'] for m in models_comparison.values()]
params = [m['params'] for m in models_comparison.values()]
times = [m['time'] for m in models_comparison.values()]

# Accuracy
axes[0].bar(models, accs, color=['steelblue', 'green'], alpha=0.7)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Test Accuracy', fontsize=12, fontweight='bold')
axes[0].set_ylim(0, 1)
axes[0].grid(axis='y', alpha=0.3)

for i, acc in enumerate(accs):
    axes[0].text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontweight='bold')

# Parameters
axes[1].bar(models, params, color=['steelblue', 'green'], alpha=0.7)
axes[1].set_ylabel('Parameters', fontsize=12)
axes[1].set_title('Model Size', fontsize=12, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

for i, p in enumerate(params):
    axes[1].text(i, p + 50000, f'{p/1e6:.1f}M', ha='center', fontweight='bold')

# Training time
axes[2].bar(models, times, color=['steelblue', 'green'], alpha=0.7)
axes[2].set_ylabel('Time (seconds)', fontsize=12)
axes[2].set_title('Training Time', fontsize=12, fontweight='bold')
axes[2].grid(axis='y', alpha=0.3)

for i, t in enumerate(times):
    axes[2].text(i, t + 1, f'{t:.1f}s', ha='center', fontweight='bold')

plt.suptitle('EfficientNet vs Baseline CNN', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('efficientnet_comparison.png', dpi=150)
print("✅ Comparação salva: efficientnet_comparison.png")

print("\n💡 EFFICIENTNET:")
print("  • Compound Scaling: depth + width + resolution")
print("  • B0-B7: 7 variantes (trade-off accuracy/speed)")
print("  • Mobile-friendly: Otimizado para dispositivos")
print("  • AutoML: Neural Architecture Search")

print("\n🎯 COMPOUND SCALING:")
print("  • Depth (d): Mais camadas")
print("  • Width (w): Mais filtros por camada")
print("  • Resolution (r): Maior resolução de entrada")
print("  • Scaling: d^α × w^β × r^γ (α+β+2γ≈2)")

print("\n🏆 RESULTADOS IMAGENET:")
print("  • EfficientNet-B0: 77.1% top-1 (5.3M params)")
print("  • EfficientNet-B7: 84.3% top-1 (66M params)")
print("  • 10x menos parâmetros que ResNet")
print("  • 8.4x mais rápido que GPipe")

print("\n✅ EFFICIENTNET COMPLETO!")
