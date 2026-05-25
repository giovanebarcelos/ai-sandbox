# GO1248-Tensorflow
from tensorflow.keras.applications import ResNet50
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass


if __name__ == '__main__':
    base = ResNet50(weights='imagenet', include_top=False)
    base.trainable = False  # Congelar

    # ─── VISUALIZAÇÃO: RESNET50 COMO EXTRATOR DE FEATURES ───
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Diagrama: camadas congeladas vs treináveis
    ax = axes[0]
    ax.set_xlim(0, 8)
    ax.set_ylim(-0.5, 11)
    ax.axis('off')
    ax.set_title('ResNet50: Feature Extractor\n(camadas congeladas vs treináveis)', fontsize=12, fontweight='bold')

    blocos_resnet = [
        ('Input\n224×224×3',   '#76b7b2', True),
        ('Bloco 1\n(stem)',       '#aaaaaa', True),
        ('Stage 1\n(3 blocos)',   '#aaaaaa', True),
        ('Stage 2\n(4 blocos)',   '#aaaaaa', True),
        ('Stage 3\n(6 blocos)',   '#aaaaaa', True),
        ('Stage 4\n(3 blocos)',   '#aaaaaa', True),
        ('GAP\n2048-dim',         '#f28e2b', False),
        ('Dense(256)\n+ ReLU',    '#59a14f', False),
        ('Dropout(0.5)',           '#edc948', False),
        ('Dense(N)\nSoftmax',     '#e15759', False),
    ]

    for i, (nome, cor, frozen) in enumerate(blocos_resnet):
        y = 10 - i
        rect = mpatches.FancyBboxPatch((1.5, y - 0.38), 3, 0.72,
                                        boxstyle='round,pad=0.05', linewidth=1.5,
                                        edgecolor='black', facecolor=cor, alpha=0.8)
        ax.add_patch(rect)
        ax.text(3, y, nome, ha='center', va='center', fontsize=9, fontweight='bold',
                color='white' if cor not in ['#aaaaaa', '#76b7b2'] else 'black')
        status = '❌ Congelado' if frozen else '✅ Treinável'
        ax.text(5, y, status, ha='left', va='center', fontsize=9,
                color='#888' if frozen else '#59a14f', fontweight='bold')
        if i < len(blocos_resnet) - 1:
            ax.annotate('', xy=(3, 10 - (i+1) + 0.38), xytext=(3, y - 0.38),
                        arrowprops=dict(arrowstyle='->', color='#555', lw=1.3))

    # Barras: parâmetros treináveis vs congelados
    ax2 = axes[1]
    categorias = ['Congelados\n(ResNet50)\n~23.5M', 'Treináveis\n(Classificador)\n~526K']
    valores = [23.5, 0.526]
    colors_bar = ['#aaaaaa', '#59a14f']
    bars = ax2.bar(categorias, valores, color=colors_bar, edgecolor='black', width=0.4)
    ax2.set_ylabel('Parâmetros (Milhões)')
    ax2.set_title('ResNet50: Parâmetros Congelados vs Treináveis\n(~97.8% congelados!)', fontsize=12)
    for bar, v in zip(bars, valores):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f'{v:.2f}M', ha='center', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    total = sum(valores)
    ax2.text(0.5, 12, f'Total: {total:.2f}M | Treináveis: {valores[1]/total*100:.1f}%',
             ha='center', transform=ax2.transAxes, fontsize=10, color='navy')

    plt.suptitle('ResNet50 como Feature Extractor (Transfer Learning)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
