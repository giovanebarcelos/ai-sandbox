# GO1254-ExtrairFiltrosDa
# Extrair filtros da primeira camada
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass


if __name__ == '__main__':
    from tensorflow import keras
    from tensorflow.keras.models import Model

    # Treinar modelo rápido para extrair filtros
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255

    model = keras.Sequential([
        keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1), name='conv1'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, (3,3), activation='relu', name='conv2'),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(10, activation='softmax'),
    ])
    model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train[:2000], y_train[:2000], epochs=2, verbose=0)

    # get_weights(): retorna [pesos, biases] da camada
    # filters.shape = (kernel_h, kernel_w, in_channels, n_filters)
    # Para conv1 com 32 filtros 3×3 grayscale: (3, 3, 1, 32)
    filters, biases = model.layers[0].get_weights()
    print(f"Shape: {filters.shape}")  # (3, 3, 1, 32)

    # Extrair feature maps usando Model secundário
    # Filtra apenas camadas de nome 'conv' (ignora pooling, dense etc.)
    layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
    # Model funcional com múltiplas saídas: 1 entrada, N saídas de ativação
    feature_model = Model(inputs=model.input, outputs=layer_outputs)
    image = x_train[0:1]
    # feature_maps[i] = ativação da i-ésima camada conv para 'image'
    feature_maps = feature_model.predict(image, verbose=0)

    # ─── VISUALIZAÇÃO: FILTROS E FEATURE MAPS ───
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Filtros normalizados da conv1
    f_min, f_max = filters.min(), filters.max()
    filters_norm = (filters - f_min) / (f_max - f_min + 1e-8)
    n_show = 16
    axes[0, 0].axis('off')
    axes[0, 0].set_title(f'Filtros da Conv1 (primeiros {n_show} de 32)', fontsize=11, fontweight='bold')
    for idx in range(n_show):
        r, c = divmod(idx, 8)
        sub = fig.add_axes([0.05 + c * 0.055, 0.55 - r * 0.2, 0.048, 0.17])
        sub.imshow(filters_norm[:, :, 0, idx], cmap='gray')
        sub.set_title(f'F{idx+1}', fontsize=6)
        sub.axis('off')

    # Feature maps da conv1 (primeiros 16)
    fm1 = feature_maps[0]
    axes[0, 1].axis('off')
    axes[0, 1].set_title(f'Feature Maps Conv1 — dígito: {y_train[0]}', fontsize=11, fontweight='bold')
    for idx in range(n_show):
        r, c = divmod(idx, 8)
        sub = fig.add_axes([0.53 + c * 0.055, 0.55 - r * 0.2, 0.048, 0.17])
        sub.imshow(fm1[0, :, :, idx], cmap='viridis')
        sub.axis('off')

    # Imagem original
    axes[1, 0].imshow(image[0, :, :, 0], cmap='gray')
    axes[1, 0].set_title(f'Imagem Original — Dígito: {y_train[0]}', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')

    # Ativação média por filtro
    mean_acts = [fm1[0, :, :, i].mean() for i in range(32)]
    axes[1, 1].bar(range(32), mean_acts, color=plt.cm.viridis(np.linspace(0, 1, 32)), edgecolor='black')
    axes[1, 1].set_xlabel('Filtro')
    axes[1, 1].set_ylabel('Ativação Média')
    axes[1, 1].set_title('Ativação Média por Filtro (Conv1)\nFiltros mais "ativos" para este dígito', fontsize=11)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Extração e Visualização de Filtros e Feature Maps', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

