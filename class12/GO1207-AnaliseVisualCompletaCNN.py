# GO1207-AnaliseVisualCompletaCNN
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Carregar modelo treinado
model = keras.models.load_model('best_cifar10_model.h5')

# 1. VISUALIZAR ARQUITETURA
def plot_model_architecture(model):
    """
    Plota arquitetura detalhada da CNN
    """
    print("📐 Arquitetura da Rede:")
    print("="*60)

    for i, layer in enumerate(model.layers):
        layer_type = layer.__class__.__name__
        if hasattr(layer, 'output_shape'):
            output_shape = layer.output_shape
        else:
            output_shape = 'N/A'

        # Contar parâmetros
        params = layer.count_params()

        print(f"{i+1:2d}. {layer.name:15s} | {layer_type:20s} | "
              f"Shape: {str(output_shape):25s} | Params: {params:,}")

    print("="*60)

plot_model_architecture(model)

# 2. VISUALIZAR FILTROS DA PRIMEIRA CAMADA
def visualize_conv_filters(model, layer_name='conv1', num_filters=32):
    """
    Visualiza filtros de uma camada convolucional
    """
    # Obter camada
    layer = model.get_layer(layer_name)
    filters, biases = layer.get_weights()

    print(f"\n🔍 Filtros da camada '{layer_name}':")
    print(f"   Shape: {filters.shape}")
    print(f"   {filters.shape[3]} filtros de {filters.shape[0]}x{filters.shape[1]}x{filters.shape[2]}")

    # Normalizar filtros para visualização
    f_min, f_max = filters.min(), filters.max()
    filters_normalized = (filters - f_min) / (f_max - f_min)

    # Plotar filtros
    n_cols = 8
    n_rows = min(num_filters, filters.shape[3]) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 2*n_rows))
    axes = axes.ravel()

    for i in range(min(num_filters, filters.shape[3])):
        # Obter filtro (3x3x3) e converter para RGB
        f = filters_normalized[:, :, :, i]

        # Se tiver 3 canais, mostrar RGB; senão, converter pra grayscale
        if f.shape[2] == 3:
            axes[i].imshow(f)
        else:
            axes[i].imshow(f[:, :, 0], cmap='viridis')

        axes[i].set_title(f'F{i}', fontsize=8)
        axes[i].axis('off')

    plt.suptitle(f'Filtros Aprendidos - Camada {layer_name}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'filters_{layer_name}.png', dpi=150)
    print(f"✅ Filtros salvos em filters_{layer_name}.png")

visualize_conv_filters(model, 'conv1', 32)

# 3. FEATURE MAPS DE UMA IMAGEM
def visualize_feature_maps(model, img, layers_to_show=None):
    """
    Visualiza feature maps de camadas específicas
    """
    if layers_to_show is None:
        # Mostrar todas as camadas Conv2D
        layers_to_show = [layer.name for layer in model.layers 
                         if 'conv' in layer.name.lower()]

    # Criar modelos intermediários
    outputs = [model.get_layer(name).output for name in layers_to_show]
    feature_model = keras.Model(inputs=model.input, outputs=outputs)

    # Obter feature maps
    img_batch = np.expand_dims(img, axis=0)
    feature_maps = feature_model.predict(img_batch, verbose=0)

    # Plotar
    num_layers = len(layers_to_show)
    fig = plt.figure(figsize=(20, 4*num_layers))
    gs = GridSpec(num_layers, 1, figure=fig)

    for i, (layer_name, fmaps) in enumerate(zip(layers_to_show, feature_maps)):
        # Pegar apenas primeiros 8 feature maps
        num_features = min(8, fmaps.shape[-1])

        ax_main = fig.add_subplot(gs[i, :])
        ax_main.axis('off')
        ax_main.set_title(f'Layer: {layer_name} | Shape: {fmaps.shape}', 
                         fontsize=12, fontweight='bold', loc='left')

        # Criar subgrid para feature maps
        for j in range(num_features):
            ax = plt.subplot(num_layers, num_features, i*num_features + j + 1)
            fmap = fmaps[0, :, :, j]
            ax.imshow(fmap, cmap='viridis')
            ax.set_title(f'FM{j}', fontsize=8)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('feature_maps_progression.png', dpi=150, bbox_inches='tight')
    print("✅ Feature maps salvos em feature_maps_progression.png")

# Carregar uma imagem de teste
(X_train, _), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_test = X_test.astype('float32') / 255.0

# Escolher uma imagem
test_img = X_test[42]  # Pode mudar o índice

# Mostrar imagem original
plt.figure(figsize=(3, 3))
plt.imshow(test_img)
plt.title('Imagem Original')
plt.axis('off')
plt.savefig('original_image.png', dpi=150)

# Visualizar feature maps
layers_to_analyze = ['conv1', 'conv3', 'conv5']
visualize_feature_maps(model, test_img, layers_to_analyze)

# 4. ACTIVATION MAXIMIZATION
def generate_pattern(model, layer_name, filter_index, steps=50, learning_rate=1.0):
    """
    Gera padrão que maximiza ativação de um filtro específico
    """
    # Criar modelo que retorna ativação do filtro
    layer = model.get_layer(layer_name)
    feature_extractor = keras.Model(inputs=model.input, outputs=layer.output)

    # Inicializar com ruído
    img = tf.random.uniform((1, 32, 32, 3), minval=0.4, maxval=0.6)

    for step in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(img)
            activation = feature_extractor(img)

            # Perda = negativo da ativação média do filtro
            loss = -tf.reduce_mean(activation[:, :, :, filter_index])

        # Calcular gradiente
        grads = tape.gradient(loss, img)

        # Normalizar gradiente
        grads = tf.math.l2_normalize(grads)

        # Atualizar imagem
        img += grads * learning_rate

        # Clipar valores para [0, 1]
        img = tf.clip_by_value(img, 0, 1)

    return img[0].numpy()

# Gerar padrões para alguns filtros
print("\n🎨 Gerando padrões que maximizam ativação...")
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.ravel()

for i in range(8):
    pattern = generate_pattern(model, 'conv1', filter_index=i, steps=50)
    axes[i].imshow(pattern)
    axes[i].set_title(f'Filtro {i}', fontsize=10)
    axes[i].axis('off')

plt.suptitle('Padrões que Maximizam Ativação dos Filtros', 
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('activation_maximization.png', dpi=150)
print("✅ Padrões salvos em activation_maximization.png")

# 5. ANÁLISE DE GRADIENTES
def analyze_gradients(model, img, class_index):
    """
    Analisa gradientes da imagem em relação à classe predita
    """
    img_batch = np.expand_dims(img, axis=0)

    with tf.GradientTape() as tape:
        img_tensor = tf.Variable(img_batch, dtype=tf.float32)
        predictions = model(img_tensor)
        target_class = predictions[:, class_index]

    # Gradientes
    gradients = tape.gradient(target_class, img_tensor)

    return gradients[0].numpy()

# Exemplo
test_img = X_test[100]
pred = model.predict(np.expand_dims(test_img, axis=0), verbose=0)
pred_class = np.argmax(pred[0])

grads = analyze_gradients(model, test_img, pred_class)

# Visualizar
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(test_img)
axes[0].set_title('Imagem Original', fontsize=12)
axes[0].axis('off')

# Gradientes (abs)
grad_vis = np.abs(grads).mean(axis=2)
axes[1].imshow(grad_vis, cmap='hot')
axes[1].set_title('Gradientes (Importância)', fontsize=12)
axes[1].axis('off')

# Sobreposição
axes[2].imshow(test_img)
axes[2].imshow(grad_vis, cmap='hot', alpha=0.5)
axes[2].set_title('Sobreposição', fontsize=12)
axes[2].axis('off')

plt.suptitle('Análise de Gradientes - Quais pixels influenciam a predição?', 
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('gradient_analysis.png', dpi=150)
print("✅ Análise de gradientes salva")

print("\n✅ ANÁLISE VISUAL COMPLETA!")
print("📁 Arquivos gerados:")
print("   - filters_conv1.png")
print("   - feature_maps_progression.png")
print("   - activation_maximization.png")
print("   - gradient_analysis.png")
