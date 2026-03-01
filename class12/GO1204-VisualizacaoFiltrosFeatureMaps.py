# GO1204-VisualizacaoFiltrosFeatureMaps
# ═══════════════════════════════════════════════════════════════════
# VISUALIZAÇÃO DE FILTROS E FEATURE MAPS DE CNNs
# Entendendo o que cada camada aprende
# ═══════════════════════════════════════════════════════════════════

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# ─── 1. CARREGAR MODELO TREINADO (usar o da Slide 11A) ───
# Se não tiver o modelo salvo, criar um simples
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255

# Criar modelo simples
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv1'),
    keras.layers.MaxPooling2D((2, 2), name='pool1'),
    keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
    keras.layers.MaxPooling2D((2, 2), name='pool2'),
    keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv3'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train[:5000], y_train[:5000], epochs=3, verbose=0)  # Treino rápido

print("Modelo treinado!")
model.summary()

# ─── 2. VISUALIZAR FILTROS (KERNELS) DA PRIMEIRA CAMADA ───
# Filtros da primeira camada Conv2D
filters, biases = model.layers[0].get_weights()
print(f"\nFiltros shape: {filters.shape}")  # (3, 3, 1, 32) = 32 filtros 3x3

# Normalizar filtros para visualização
f_min, f_max = filters.min(), filters.max()
filters_norm = (filters - f_min) / (f_max - f_min)

# Plotar todos os 32 filtros
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for i, ax in enumerate(axes.flat):
    if i < 32:
        # Cada filtro é 3x3x1
        filter_img = filters_norm[:, :, 0, i]
        ax.imshow(filter_img, cmap='gray')
        ax.set_title(f'Filter {i+1}', fontsize=10)
    ax.axis('off')

plt.suptitle('Filtros da Primeira Camada Conv (3x3)', fontsize=16)
plt.tight_layout()
plt.show()

# ─── 3. VISUALIZAR FEATURE MAPS (ATIVAÇÕES) ───
# Pegar uma imagem de teste
test_img = x_test[0:1]  # Shape: (1, 28, 28, 1)
test_img_norm = test_img.astype('float32') / 255

# Criar modelos intermediários para cada camada Conv
layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
feature_map_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)

# Gerar feature maps
feature_maps = feature_map_model.predict(test_img_norm)

print(f"\nNúmero de camadas Conv: {len(feature_maps)}")
for i, fm in enumerate(feature_maps):
    print(f"Camada {i+1}: {fm.shape}")

# ─── 4. PLOTAR FEATURE MAPS DE CADA CAMADA ───
layer_names = ['conv1', 'conv2', 'conv3']

for layer_name, feature_map in zip(layer_names, feature_maps):
    # feature_map shape: (1, height, width, channels)
    n_features = feature_map.shape[-1]  # Número de filtros

    # Plotar primeiros 16 feature maps
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for i, ax in enumerate(axes.flat):
        if i < n_features and i < 16:
            # Extrair feature map individual
            fm = feature_map[0, :, :, i]
            ax.imshow(fm, cmap='viridis')
            ax.set_title(f'Filter {i+1}', fontsize=10)
        ax.axis('off')

    plt.suptitle(f'Feature Maps - {layer_name} ({feature_map.shape[1]}x{feature_map.shape[2]}x{n_features})', 
                 fontsize=16)
    plt.tight_layout()
    plt.show()

# ─── 5. MOSTRAR IMAGEM ORIGINAL ───
plt.figure(figsize=(5, 5))
plt.imshow(test_img[0].reshape(28, 28), cmap='gray')
plt.title(f'Imagem Original - Label: {y_test[0]}', fontsize=14)
plt.axis('off')
plt.show()

# ─── 6. ANÁLISE: O QUE CADA CAMADA APRENDE? ───
print("\n" + "="*70)
print("📊 ANÁLISE: O QUE CADA CAMADA APRENDE")
print("="*70)
print("CONV1 (primeira camada):")
print("  • Detecta features simples: bordas, linhas, curvas")
print("  • Filtros respondem a orientações específicas")
print("  • Exemplo: bordas verticais, horizontais, diagonais")
print()
print("CONV2 (segunda camada):")
print("  • Combina features simples em padrões mais complexos")
print("  • Detecta cantos, texturas, formas básicas")
print("  • Maior receptive field que Conv1")
print()
print("CONV3 (terceira camada):")
print("  • Features ainda mais abstratas")
print("  • Detecta partes de objetos (para dígitos: loops, hastes)")
print("  • Representação de alto nível")
print("="*70)

# ═══════════════════════════════════════════════════════════════════
# OBSERVAÇÕES:
# • Camadas iniciais: features simples (bordas, texturas)
# • Camadas profundas: features complexas (partes, objetos)
# • Feature maps ficam menores e mais abstratos conforme avançamos
# • Visualização ajuda a entender e debugar a rede
# ═══════════════════════════════════════════════════════════════════
