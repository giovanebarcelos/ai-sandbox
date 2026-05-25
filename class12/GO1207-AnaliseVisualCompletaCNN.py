# GO1207-AnaliseVisualCompletaCNN
import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib.gridspec import GridSpec

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

# Carrega modelo previamente treinado e salvo em disco (.h5 = formato HDF5 do Keras)
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

    # layer.count_params(): soma todos os pesos (filtros + bias) da camada
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
    layer = model.get_layer(layer_name)    # get_weights() retorna [filtros, bias]; filtros shape: (kH, kW, in_ch, out_ch)    filters, biases = layer.get_weights()

    print(f"\n🔍 Filtros da camada '{layer_name}':")
    print(f"   Shape: {filters.shape}")
    print(f"   {filters.shape[3]} filtros de {filters.shape[0]}x{filters.shape[1]}x{filters.shape[2]}")

    # Normalização min-max para mapear valores dos filtros para [0,1] visível como imagem
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

        # Filtros RGB: exibir como imagem colorida (3 canais) para visualizar resposta por cor
        if f.shape[2] == 3:
            axes[i].imshow(f)
        else:
            # Canal único: mapa de calor viridis para destacar áreas de alta ativação
            axes[i].imshow(f[:, :, 0], cmap='viridis')

        axes[i].set_title(f'F{i}', fontsize=8)
        axes[i].axis('off')

    plt.suptitle(f'Filtros Aprendidos - Camada {layer_name}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
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

    # Modelo multi-saída: extrai ativações de várias camadas numa única passagem (mais eficiente)
    outputs = [model.get_layer(name).output for name in layers_to_show]
    feature_model = keras.Model(inputs=model.input, outputs=outputs)

    # np.expand_dims: adiciona dimensão de batch (32,32,3) → (1,32,32,3)
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
    plt.show()
    print("✅ Feature maps salvos em feature_maps_progression.png")

# Carregar uma imagem de teste
(X_train, _), (X_test, y_test) = keras.datasets.cifar10.load_data()
# Normalização: converte uint8 [0,255] para float32 [0,1] para entrada no modelo
X_test = X_test.astype('float32') / 255.0

# Índice 42 escolhido arbitrariamente — pode ser alterado para explorar outras imagens
test_img = X_test[42]  # Pode mudar o índice

# Mostrar imagem original
plt.figure(figsize=(3, 3))
plt.imshow(test_img)
plt.title('Imagem Original')
plt.axis('off')
plt.show()

# Visualizar feature maps
layers_to_analyze = ['conv1', 'conv3', 'conv5']
visualize_feature_maps(model, test_img, layers_to_analyze)

# 4. ACTIVATION MAXIMIZATION
def generate_pattern(model, layer_name, filter_index, steps=50, learning_rate=1.0):
    """
    Gera padrão que maximiza ativação de um filtro específico
    Técnica: gradient ascent no espaço de pixels (otimiza a IMAGEM, não os pesos)
    """
    # Criar modelo que retorna ativação do filtro
    layer = model.get_layer(layer_name)
    feature_extractor = keras.Model(inputs=model.input, outputs=layer.output)

    # Inicializar com ruído aleatório ao redor de 0.5 — ponto de partida neutro no espaço de pixels
    img = tf.random.uniform((1, 32, 32, 3), minval=0.4, maxval=0.6)

    for step in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(img)  # Observar a imagem (não os pesos) para calcular gradiente
            activation = feature_extractor(img)

            # Perda negativa: maximizar ativação média do filtro = minimizar seu negativo
            loss = -tf.reduce_mean(activation[:, :, :, filter_index])

        # Passo 1: gradiente de perda em relação à imagem (direção de aumento da ativação)
        grads = tape.gradient(loss, img)

        # Passo 2: normalização L2 do gradiente — mantém magnitude estável entre iterações
        grads = tf.math.l2_normalize(grads)

        # Passo 3: gradient ascent — atualiza a imagem na direção que maximiza a ativação
        img += grads * learning_rate

        # Passo 4: clipar para manter pixels no intervalo válido [0,1]
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
plt.show()
print("✅ Padrões salvos em activation_maximization.png")

# 5. ANÁLISE DE GRADIENTES
def analyze_gradients(model, img, class_index):
    """
    Analisa gradientes da imagem em relação à classe predita
    """
    img_batch = np.expand_dims(img, axis=0)

    with tf.GradientTape() as tape:
        # tf.Variable necessário: GradientTape só rastreia Variables por padrão (não tensores comuns)
        img_tensor = tf.Variable(img_batch, dtype=tf.float32)
        predictions = model(img_tensor)
        target_class = predictions[:, class_index]

    # Gradiente da probabilidade da classe em relação a CADA PIXEL da imagem
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

# np.abs(): valor absoluto — tanto gradientes positivos quanto negativos indicam influência
# .mean(axis=2): média sobre os 3 canais RGB para obter mapa de importância 2D
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
plt.show()
print("✅ Análise de gradientes salva")

print("\n✅ ANÁLISE VISUAL COMPLETA!")
print("📁 Arquivos gerados:")
print("   - filters_conv1.png")
print("   - feature_maps_progression.png")
print("   - activation_maximization.png")
print("   - gradient_analysis.png")
