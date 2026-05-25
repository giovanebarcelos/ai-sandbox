# GO1239-Vgg16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

def VGG16(input_shape=(224, 224, 3), num_classes=1000):
    """
    VGG-16 original (sem Batch Normalization) — Simonyan & Zisserman, 2014
    138 milhões de parâmetros totais:
      - Blocos conv 1-5: ~14.7M parâmetros
      - Camadas FC (fc1+fc2+pred): ~123.6M parâmetros (89% do total!)
    Inovacão: empilhar convoluções 3×3 pequenas equivale a campos receptivos maiores
      com menos parâmetros e mais não-linearidades
    """
    model = Sequential()

    # Bloco 1: 2 conv 3×3 com 64 filtros — input 224×224 → após pool: 112×112
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                     input_shape=input_shape, name='block1_conv1'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))  # 224→112

    # Bloco 2: 2 conv 3×3 com 128 filtros — 112×112 → 56×56
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))  # 112→56

    # Bloco 3: 3 conv 3×3 com 256 filtros — 56×56 → 28×28
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))  # 56→28

    # Bloco 4: 3 conv 3×3 com 512 filtros — 28×28 → 14×14
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))  # 28→14

    # Bloco 5: 3 conv 3×3 com 512 filtros — 14×14 → 7×7
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))   # 14→7

    # Classificador FC: 7×7×512 = 25.088 → 4096 → 4096 → num_classes
    # GARGALO: estas 3 camadas Dense contêm ~123M dos 138M parâmetros do VGG!
    # Motivação para GlobalAveragePooling em arquiteturas modernas (ResNet, Inception)
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='fc1'))   # 25088 × 4096 = ~103M
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='fc2'))   # 4096 × 4096 = ~16M
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax', name='predictions'))

    return model

# Criar e visualizar


if __name__ == "__main__":
    model = VGG16()
    model.summary()

    # Output:
    # Total params: 138,357,544
    # Trainable params: 138,357,544
    # ⚠️ 138M parâmetros = ~550MB em disco!

    # ─── VISUALIZAÇÃO: PARÂMETROS POR BLOCO ───
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
    except NameError:
        pass

    # Parâmetros por bloco do VGG-16
    blocos = ['Bloco 1\n(64 filtros)', 'Bloco 2\n(128 filtros)',
              'Bloco 3\n(256 filtros)', 'Bloco 4\n(512 filtros)',
              'Bloco 5\n(512 filtros)', 'FC1\n(4096)', 'FC2\n(4096)', 'Output\n(1000)']
    params_blocos = [38720, 221440, 1475584, 5899264, 7079936, 102764544, 16781312, 4097000]
    colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1', '#ff9da7']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Barras de parâmetros
    bars = ax1.bar(blocos, [p / 1e6 for p in params_blocos], color=colors, edgecolor='black')
    ax1.set_ylabel('Parâmetros (Milhões)')
    ax1.set_title('VGG-16: Parâmetros por Bloco (~138M total)', fontsize=12)
    for bar, p in zip(bars, params_blocos):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{p/1e6:.1f}M', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', labelsize=8)

    # Pizza da distribuição de parâmetros
    labels_pizza = [f'Bloco {i+1}' if i < 5 else f'FC{i-4}' if i < 7 else 'Out'
                    for i in range(len(blocos))]
    wedges, texts, autotexts = ax2.pie(params_blocos, labels=labels_pizza, colors=colors,
                                        autopct='%1.1f%%', startangle=90,
                                        textprops={'fontsize': 9})
    for at in autotexts:
        at.set_fontsize(8)
    ax2.set_title('VGG-16: Distribuição de Parâmetros\n(~74% nas camadas FC!)', fontsize=12)

    plt.suptitle('VGG-16 - Arquitetura (2014) | 138M Parâmetros', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # ─── VISUALIZAÇÃO: PROFUNDIDADE E FILTROS POR BLOCO ───
    fig, ax = plt.subplots(figsize=(12, 4))
    blocos_short = ['B1', 'B2', 'B3', 'B4', 'B5']
    num_convs = [2, 2, 3, 3, 3]
    num_filters = [64, 128, 256, 512, 512]
    x = np.arange(5)
    bar1 = ax.bar(x - 0.2, num_filters, 0.35, label='Nº de Filtros', color='#4e79a7', edgecolor='black')
    bar2 = ax.bar(x + 0.2, [n * 50 for n in num_convs], 0.35, label='Nº Conv × 50', color='#f28e2b', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Bloco {i+1}\n({f} filtros, {c} convs)'
                        for i, (f, c) in enumerate(zip(num_filters, num_convs))], fontsize=9)
    ax.set_ylabel('Valor')
    ax.set_title('VGG-16: Profundidade e Número de Filtros por Bloco', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
