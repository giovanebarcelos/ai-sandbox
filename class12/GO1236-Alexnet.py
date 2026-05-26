# GO1236-Alexnet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

def AlexNet(input_shape=(224, 224, 3), num_classes=1000):
    """
    AlexNet adaptado para Keras (versão simplificada)
    Original usava 2 GPUs - aqui unificamos
    Arquitetura original (Krizhevsky, 2012) - venceu ImageNet com 15.3% top-5 error
    """
    model = Sequential([
        # CONV1: 96 filtros 11×11, stride 4 — campo receptivo grande para capturar estrutura global
        # Input 224×224 → output (224-11)/4+1 = 54×54 → após pool: 26×26
        Conv2D(96, (11, 11), strides=4, activation='relu',
               input_shape=input_shape, name='conv1'),
        MaxPooling2D((3, 3), strides=2, name='pool1'),  # 26×26 → 12×12

        # CONV2: 256 filtros 5×5 — padding='same' mantém dimensão
        Conv2D(256, (5, 5), padding='same', activation='relu', name='conv2'),
        MaxPooling2D((3, 3), strides=2, name='pool2'),  # 12×12 → 5×5

        # CONV3-5: 3 convoluções consecutivas sem pooling entre elas
        # Features cada vez mais abstratas e específicas
        Conv2D(384, (3, 3), padding='same', activation='relu', name='conv3'),
        Conv2D(384, (3, 3), padding='same', activation='relu', name='conv4'),
        Conv2D(256, (3, 3), padding='same', activation='relu', name='conv5'),
        MaxPooling2D((3, 3), strides=2, name='pool3'),  # 5×5 → 2×2

        # FC Layers: 4096 → 4096 → num_classes
        # As 2 camadas FC contêm ~58M parâmetros — 93% do total!
        # Gargalo: por isso VGG/ResNet substituíram por GlobalAveragePooling
        Flatten(),
        Dense(4096, activation='relu', name='fc6'),
        Dropout(0.5),  # Dropout 50% nas FC: inovador em 2012, agora padrão
        Dense(4096, activation='relu', name='fc7'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax', name='fc8')  # 1000 classes ImageNet
    ])  # end Sequential
    return model
# Criar modelo


if __name__ == "__main__":
    model = AlexNet(num_classes=10)  # adaptado para 10 classes (CIFAR-10)

    # SGD: otimizador usado no paper original (melhor generalização que Adam em CNNs)
    model.compile(
        optimizer='sgd',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    # Total params: ~62M
    # FC6 + FC7: ~58M parâmetros (93% do modelo está nas camadas Dense!)
    # Demonstra o problema das FC layers — motivou Global Average Pooling

    # ─── VISUALIZAÇÃO: PARÂMETROS POR CAMADA ───
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
    except NameError:
        pass

    layer_names = ['conv1\n(11×11, 96)', 'pool1', 'conv2\n(5×5, 256)', 'pool2',
                   'conv3\n(3×3, 384)', 'conv4\n(3×3, 384)', 'conv5\n(3×3, 256)',
                   'pool3', 'flatten', 'fc6\n(4096)', 'drop', 'fc7\n(4096)', 'drop', 'fc8\n(10)']
    params_per_layer = [34944, 0, 614656, 0, 885120, 1327488, 884992, 0, 0, 37752832, 0, 16781312, 0, 40970]
    colors = ['#4e79a7' if p > 0 else '#cccccc' for p in params_per_layer]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Parâmetros por camada
    non_zero = [(n, p, c) for n, p, c in zip(layer_names, params_per_layer, colors) if p > 0]
    names_nz, params_nz, colors_nz = zip(*non_zero)
    bars = ax1.bar(range(len(names_nz)), [p / 1e6 for p in params_nz], color=colors_nz, edgecolor='black')
    ax1.set_xticks(range(len(names_nz)))
    ax1.set_xticklabels(names_nz, rotation=30, ha='right', fontsize=8)
    ax1.set_ylabel('Parâmetros (Milhões)')
    ax1.set_title('AlexNet: Parâmetros por Camada (~62M total)', fontsize=12)
    for bar, p in zip(bars, params_nz):
        if p > 1e6:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     f'{p/1e6:.1f}M', ha='center', va='bottom', fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')

    # Evolução do tamanho das feature maps
    etapas = ['Input\n227×227×3', 'Conv1\n55×55×96', 'Pool1\n27×27×96',
              'Conv2\n27×27×256', 'Pool2\n13×13×256', 'Conv3\n13×13×384',
              'Conv4\n13×13×384', 'Conv5\n13×13×256', 'Pool3\n6×6×256',
              'FC6\n4096', 'FC7\n4096', 'FC8\n1000']
    volumes = [227*227*3, 55*55*96, 27*27*96, 27*27*256, 13*13*256,
               13*13*384, 13*13*384, 13*13*256, 6*6*256, 4096, 4096, 1000]
    ax2.plot(range(len(volumes)), volumes, 'o-', color='#e15759', linewidth=2, markersize=7)
    ax2.fill_between(range(len(volumes)), volumes, alpha=0.15, color='#e15759')
    ax2.set_xticks(range(len(etapas)))
    ax2.set_xticklabels(etapas, rotation=45, ha='right', fontsize=7)
    ax2.set_ylabel('Volume (número de elementos)')
    ax2.set_title('AlexNet: Evolução do Volume por Camada', fontsize=12)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('AlexNet - Arquitetura (2012)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
