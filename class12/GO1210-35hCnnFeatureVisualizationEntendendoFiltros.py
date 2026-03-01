# GO1210-35hCnnFeatureVisualizationEntendendoFiltros
# ═══════════════════════════════════════════════════════════════════
# CNN FEATURE VISUALIZATION - ENTENDENDO FILTROS
# Visualizar o que cada camada da CNN aprende
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

print("🔍 CNN FEATURE VISUALIZATION")
print("=" * 70)

# ─── 1. TREINAR CNN SIMPLES ───
print("\n🏗️ Treinando CNN simples no MNIST...")

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preparar dados
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Reduzir para treino rápido
X_train = X_train[:5000]
y_train = y_train[:5000]

model = Sequential([
    Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1), name='conv1'),
    MaxPooling2D((2, 2), name='pool1'),
    Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2'),
    MaxPooling2D((2, 2), name='pool2'),
    Conv2D(64, (3, 3), activation='relu', padding='same', name='conv3'),
    Flatten(),
    Dense(64, activation='relu', name='dense1'),
    Dense(10, activation='softmax', name='output')
], name='MNIST_CNN')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("  Treinando (5 epochs)...")
history = model.fit(X_train, y_train, epochs=5, batch_size=128, verbose=0, validation_split=0.2)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"  Test Accuracy: {test_acc:.4f}")

# ─── 2. VISUALIZAR FILTROS DA PRIMEIRA CAMADA ───
print("\n🎨 Visualizando filtros da primeira camada...")

conv1_weights = model.get_layer('conv1').get_weights()[0]

print(f"  Shape: {conv1_weights.shape}")  # (3, 3, 1, 16)
print(f"  Número de filtros: {conv1_weights.shape[-1]}")

# Plot filtros
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
axes = axes.ravel()

for i in range(16):
    filter_img = conv1_weights[:, :, 0, i]
    axes[i].imshow(filter_img, cmap='gray')
    axes[i].set_title(f'Filter {i+1}', fontsize=9)
    axes[i].axis('off')

plt.suptitle('Filtros da Primeira Camada Conv', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('cnn_filters_conv1.png', dpi=150)
print("✅ Filtros salvos: cnn_filters_conv1.png")

# ─── 3. VISUALIZAR FEATURE MAPS ───
print("\n🗺️ Visualizando feature maps...")

# Selecionar uma imagem
test_img = X_test[0:1]  # Primeiro dígito do test
test_label = y_test[0]

print(f"  Imagem: Dígito {test_label}")

# Criar modelos intermediários para cada camada conv
layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)

# Obter activations
activations = activation_model.predict(test_img, verbose=0)

print(f"  Camadas conv: {len(activations)}")

# Visualizar
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

for layer_idx, (layer_name, activation) in enumerate(zip(['conv1', 'conv2', 'conv3'], activations)):
    num_filters = activation.shape[-1]

    # Subplot para cada layer
    ax = axes[layer_idx]

    # Criar grid de feature maps
    n_cols = 8
    n_rows = (num_filters + n_cols - 1) // n_cols

    feature_map_grid = np.zeros((n_rows * activation.shape[1], n_cols * activation.shape[2]))

    for i in range(min(num_filters, n_rows * n_cols)):
        row = i // n_cols
        col = i % n_cols

        feature_map = activation[0, :, :, i]
        feature_map_grid[
            row * activation.shape[1]:(row + 1) * activation.shape[1],
            col * activation.shape[2]:(col + 1) * activation.shape[2]
        ] = feature_map

    ax.imshow(feature_map_grid, cmap='viridis')
    ax.set_title(f'{layer_name} - {num_filters} filtros ({activation.shape[1]}x{activation.shape[2]})', 
                fontsize=12, fontweight='bold')
    ax.axis('off')

plt.suptitle(f'Feature Maps para Dígito {test_label}', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('cnn_feature_maps.png', dpi=150)
print("✅ Feature maps salvos: cnn_feature_maps.png")

# ─── 4. MAXIMALLY ACTIVATING IMAGES ───
print("\n🔥 Encontrando imagens que maximizam ativação...")

# Para cada filtro da conv1, encontrar imagens do test set que mais ativam
conv1_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('conv1').output)

conv1_activations = conv1_model.predict(X_test[:1000], verbose=0)  # 1000 imagens

# Para filtro 0, encontrar top-5 imagens
filter_idx = 0
filter_activations = conv1_activations[:, :, :, filter_idx]

# Max activation por imagem
max_activations = np.max(filter_activations, axis=(1, 2))

# Top 5 imagens
top5_indices = np.argsort(max_activations)[-5:][::-1]

print(f"  Filtro {filter_idx}: Top 5 imagens que mais ativam")

fig, axes = plt.subplots(1, 5, figsize=(15, 3))

for i, idx in enumerate(top5_indices):
    axes[i].imshow(X_test[idx].reshape(28, 28), cmap='gray')
    axes[i].set_title(f'Img {idx}\nLabel: {y_test[idx]}\nAct: {max_activations[idx]:.2f}', fontsize=9)
    axes[i].axis('off')

plt.suptitle(f'Top-5 Imagens que Maximizam Filtro {filter_idx}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('cnn_max_activating.png', dpi=150)
print("✅ Max activating salvos: cnn_max_activating.png")

# ─── 5. ANÁLISE POR CAMADA ───
print("\n" + "="*70)
print("📊 ANÁLISE DAS CAMADAS")
print("="*70)

for layer in model.layers:
    if 'conv' in layer.name or 'pool' in layer.name:
        output_shape = layer.output_shape
        if hasattr(layer, 'filters'):
            print(f"  {layer.name:10s}: Output {output_shape[1:]} | Filtros: {layer.filters}")
        else:
            print(f"  {layer.name:10s}: Output {output_shape[1:]}")

print("\n💡 O QUE CADA CAMADA APRENDE:")
print("  • CONV1 (inicial): Bordas, linhas, curvas simples")
print("  • CONV2 (meio): Texturas, padrões compostos")
print("  • CONV3 (profunda): Partes de objetos, features abstratas")
print("  • FC (final): Combinações high-level para classificação")

print("\n📚 TÉCNICAS DE VISUALIZAÇÃO:")
print("  • Filter Visualization: Ver pesos dos filtros")
print("  • Feature Maps: Ativações para input específico")
print("  • Maximally Activating: Inputs que maximizam filtro")
print("  • Grad-CAM: Heatmap de importância (já vimos!)")
print("  • Deep Dream: Amplificar padrões")

print("\n✅ FEATURE VISUALIZATION COMPLETA!")
