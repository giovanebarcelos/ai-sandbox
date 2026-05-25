# GO1236-Alexnet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

def AlexNet(input_shape=(224, 224, 3), num_classes=1000):
    """
    AlexNet adaptado para Keras (versão simplificada)
    Original usava 2 GPUs - aqui unificamos
    """
    model = Sequential([
        # CONV1: 96 filtros 11×11, stride 4
        Conv2D(96, (11, 11), strides=4, activation='relu', 
               input_shape=input_shape, name='conv1'),
        MaxPooling2D((3, 3), strides=2, name='pool1'),

        # CONV2: 256 filtros 5×5
        Conv2D(256, (5, 5), padding='same', activation='relu', name='conv2'),
        MaxPooling2D((3, 3), strides=2, name='pool2'),

        # CONV3: 384 filtros 3×3
        Conv2D(384, (3, 3), padding='same', activation='relu', name='conv3'),

        # CONV4: 384 filtros 3×3
        Conv2D(384, (3, 3), padding='same', activation='relu', name='conv4'),

        # CONV5: 256 filtros 3×3
        Conv2D(256, (3, 3), padding='same', activation='relu', name='conv5'),
        MaxPooling2D((3, 3), strides=2, name='pool3'),

        # FC Layers
        Flatten(),
        Dense(4096, activation='relu', name='fc6'),
        Dropout(0.5),
        Dense(4096, activation='relu', name='fc7'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax', name='fc8')
    ])

    return model

# Criar modelo


if __name__ == "__main__":
    model = AlexNet(num_classes=10)  # 10 classes para CIFAR-10

    # Compilar com otimizador SGD (como original)
    model.compile(
        optimizer='sgd',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Output:
    # Total params: 62,378,344
    # Trainable params: 62,378,344
    # Non-trainable params: 0

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
