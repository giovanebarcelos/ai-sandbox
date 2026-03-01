# GO1226-ComparacaoArquiteturasCNN
# ═══════════════════════════════════════════════════════════════════
# COMPARAÇÃO DE ARQUITETURAS CNN - BENCHMARK SISTEMÁTICO
# LeNet, VGG, ResNet - Trade-offs e Casos de Uso
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout,
    BatchNormalization, GlobalAveragePooling2D, Add, Input
)
from tensorflow.keras.applications import VGG16, ResNet50
import matplotlib.pyplot as plt
import time

print("🏗️ CNN ARCHITECTURE COMPARISON")
print("=" * 70)

# ─── 1. DEFINIR ARQUITETURAS ───

print("\n📐 Construindo arquiteturas...\n")

# LeNet-5 (1998) - Baseline
def build_lenet(input_shape=(28, 28, 1), num_classes=10):
    model = Sequential([
        Conv2D(6, (5, 5), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(16, (5, 5), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(120, activation='relu'),
        Dense(84, activation='relu'),
        Dense(num_classes, activation='softmax')
    ], name='LeNet5')
    return model

# Custom CNN - Médio
def build_custom_cnn(input_shape=(32, 32, 3), num_classes=10):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ], name='CustomCNN')
    return model

# Mini ResNet com Residual Blocks
def residual_block(x, filters, name=None):
    fx = Conv2D(filters, 3, padding='same', name=f'{name}_conv1')(x)
    fx = BatchNormalization(name=f'{name}_bn1')(fx)
    fx = tf.keras.layers.Activation('relu')(fx)

    fx = Conv2D(filters, 3, padding='same', name=f'{name}_conv2')(fx)
    fx = BatchNormalization(name=f'{name}_bn2')(fx)

    if x.shape[-1] != filters:
        x = Conv2D(filters, 1, name=f'{name}_shortcut')(x)

    out = Add(name=f'{name}_add')([fx, x])
    out = tf.keras.layers.Activation('relu')(out)
    return out

def build_mini_resnet(input_shape=(32, 32, 3), num_classes=10):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = residual_block(x, 32, name='res1')
    x = MaxPooling2D(2)(x)
    x = residual_block(x, 64, name='res2')
    x = MaxPooling2D(2)(x)
    x = residual_block(x, 128, name='res3')

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs, name='MiniResNet')

# ─── 2. COMPILAR E ANALISAR ───

INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10

architectures = {
    'LeNet-5': build_lenet(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES),
    'Custom CNN': build_custom_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES),
    'Mini ResNet': build_mini_resnet(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
}

# Compilar todos
for name, model in architectures.items():
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

print("✅ Todas arquiteturas compiladas\n")

# ─── 3. BENCHMARK: PARÂMETROS ───

print("="*70)
print("📊 BENCHMARK 1: COMPLEXIDADE DE MODELO")
print("="*70 + "\n")

results = []

for name, model in architectures.items():
    total_params = model.count_params()
    num_layers = len(model.layers)
    conv_layers = [l for l in model.layers if 'conv' in l.name.lower()]
    num_conv = len(conv_layers)

    results.append({
        'name': name,
        'params': total_params,
        'layers': num_layers,
        'conv_layers': num_conv
    })

    print(f"🔧 {name:18s} │ Params: {total_params:>10,} │ "
          f"Layers: {num_layers:>3} │ Conv: {num_conv:>2}")

# ─── 4. BENCHMARK: INFERENCE SPEED ───

print("\n" + "="*70)
print("⚡ BENCHMARK 2: VELOCIDADE DE INFERÊNCIA")
print("="*70 + "\n")

test_batch = np.random.rand(32, *INPUT_SHAPE).astype(np.float32)

for i, (name, model) in enumerate(architectures.items()):
    # Warmup
    _ = model.predict(test_batch[:1], verbose=0)

    # Medir tempo
    times = []
    for _ in range(10):
        start = time.time()
        _ = model.predict(test_batch, verbose=0)
        times.append(time.time() - start)

    avg_time = np.mean(times) * 1000  # ms
    throughput = 32 / (avg_time / 1000)  # images/sec

    results[i]['inference_ms'] = avg_time
    results[i]['throughput'] = throughput

    print(f"⚡ {name:18s} │ {avg_time:>6.2f} ms/batch │ {throughput:>6.1f} img/s")

# ─── 5. VISUALIZAR COMPARAÇÕES ───

print("\n📊 Gerando gráficos comparativos...\n")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

names = [r['name'] for r in results]
params = [r['params'] / 1e6 for r in results]
inference = [r['inference_ms'] for r in results]
throughput = [r['throughput'] for r in results]
layers = [r['layers'] for r in results]

# 1. Parâmetros
axes[0, 0].barh(names, params, color='steelblue')
axes[0, 0].set_xlabel('Parâmetros (Milhões)', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Complexidade do Modelo', fontsize=14, fontweight='bold')
axes[0, 0].grid(axis='x', alpha=0.3)
for i, v in enumerate(params):
    axes[0, 0].text(v + 0.1, i, f'{v:.2f}M', va='center')

# 2. Tempo de inferência
axes[0, 1].barh(names, inference, color='coral')
axes[0, 1].set_xlabel('Tempo (ms/batch)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Velocidade de Inferência', fontsize=14, fontweight='bold')
axes[0, 1].grid(axis='x', alpha=0.3)
for i, v in enumerate(inference):
    axes[0, 1].text(v + 0.5, i, f'{v:.1f}ms', va='center')

# 3. Throughput
axes[1, 0].barh(names, throughput, color='seagreen')
axes[1, 0].set_xlabel('Throughput (imagens/segundo)', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Capacidade de Processamento', fontsize=14, fontweight='bold')
axes[1, 0].grid(axis='x', alpha=0.3)
for i, v in enumerate(throughput):
    axes[1, 0].text(v + 5, i, f'{v:.0f}', va='center')

# 4. Trade-off: Params vs Speed
axes[1, 1].scatter(params, throughput, s=200, c=layers, cmap='viridis', 
                   alpha=0.7, edgecolors='black')
for i, name in enumerate(names):
    axes[1, 1].annotate(name, (params[i], throughput[i]), 
                       textcoords="offset points", xytext=(5,5), fontsize=9)
axes[1, 1].set_xlabel('Parâmetros (Milhões)', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Throughput (img/s)', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Trade-off: Complexidade vs Velocidade', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Comparação de Arquiteturas CNN - Benchmark Completo', 
            fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('architecture_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Comparação salva: architecture_comparison.png\n")

# ─── 6. RECOMENDAÇÕES ───

print("="*70)
print("💡 RECOMENDAÇÕES DE ARQUITETURA")
print("="*70 + "\n")

print("🟢 LENET-5:")
print("  Quando usar: Prototipagem, datasets pequenos (MNIST)")
print("  Prós: Rápido, poucos parâmetros")
print("  Contras: Accuracy limitada\n")

print("🔵 CUSTOM CNN:")
print("  Quando usar: Controle total, problema específico")
print("  Prós: Flexível, otimizável")
print("  Contras: Requer expertise\n")

print("🟡 MINI RESNET:")
print("  Quando usar: Datasets médios, recursos limitados")
print("  Prós: Balance entre LeNet e ResNet full")
print("  Contras: Pode não ter capacidade para ImageNet-scale\n")

print("="*70)
print("🎯 GUIA DE DECISÃO RÁPIDO")
print("="*70 + "\n")

print("  Dataset < 10k images → LeNet ou Custom CNN pequeno")
print("  Dataset 10k-100k → ResNet50 (transfer learning)")
print("  Dataset > 100k → ResNet50 ou EfficientNet (treinar do zero)")
print("  Edge/Mobile → EfficientNet ou MobileNet")
print("  GPU limitada → LeNet, Custom CNN, ou MiniResNet")
print("  Máxima accuracy → EfficientNet ou ResNet50+")

print("\n✅ COMPARAÇÃO DE ARQUITETURAS COMPLETA!")
