# GO1229-TensorflowTensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass


if __name__ == "__main__":
    model = Sequential([
        # 1ª Camada Convolucional
        Conv2D(32, (3,3), activation='relu',
               input_shape=(28, 28, 1)),
        # Output: (26, 26, 32)

        # 2ª Camada Convolucional
        Conv2D(64, (3,3), activation='relu'),
        # Output: (24, 24, 64)

        # Camada de Pooling
        MaxPooling2D(pool_size=(2, 2)),
        # Output: (12, 12, 64)

        # Achatamento para camadas densas
        Flatten(),
        # Output: 9216 neurônios

        # Camadas totalmente conectadas
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')  # 10 classes
    ])

    # ─── VISUALIZAÇÃO: DIMENSÕES DAS FEATURE MAPS ───
    camadas = ['Input\n(28×28×1)', 'Conv2D(32)\n(26×26×32)', 'Conv2D(64)\n(24×24×64)',
               'MaxPool\n(12×12×64)', 'Flatten\n(9216)', 'Dense(128)\n(128)', 'Output\n(10)']
    volumes = [28*28*1, 26*26*32, 24*24*64, 12*12*64, 9216, 128, 10]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Diagrama de fluxo com tamanhos
    colors = ['#4e79a7', '#f28e2b', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948']
    x_pos = np.arange(len(camadas))
    bars = ax1.bar(x_pos, volumes, color=colors, edgecolor='black', alpha=0.85)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(camadas, fontsize=8)
    ax1.set_ylabel('Número de Elementos (Volume)')
    ax1.set_title('Evolução do Volume por Camada', fontsize=12)
    ax1.set_yscale('log')
    for bar, vol in zip(bars, volumes):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                 f'{vol:,}', ha='center', va='bottom', fontsize=7, rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')

    # Fluxo da arquitetura
    ax2.set_xlim(0, 14)
    ax2.set_ylim(0, 6)
    ax2.axis('off')
    ax2.set_title('Fluxo da CNN para MNIST', fontsize=12)
    nomes = ['Input\n28×28×1', 'Conv\n26×26×32', 'Conv\n24×24×64', 'Pool\n12×12×64',
             'Flatten\n9216', 'Dense\n128', 'Softmax\n10']
    x_vals = [1, 3, 5, 7, 9, 11, 13]
    for i, (x, nome, cor) in enumerate(zip(x_vals, nomes, colors)):
        rect = plt.Rectangle((x - 0.6, 2.2), 1.2, 1.6, linewidth=1.5,
                              edgecolor='black', facecolor=cor, alpha=0.8)
        ax2.add_patch(rect)
        ax2.text(x, 3, nome, ha='center', va='center', fontsize=7.5, fontweight='bold')
        if i < len(x_vals) - 1:
            ax2.annotate('', xy=(x_vals[i + 1] - 0.6, 3), xytext=(x + 0.6, 3),
                         arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    plt.tight_layout()
    plt.show()
