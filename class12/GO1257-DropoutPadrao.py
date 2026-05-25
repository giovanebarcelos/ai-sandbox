# GO1257-DropoutPadrao
# Dropout padrão (após Dense)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass


if __name__ == '__main__':
    # Dropout padrão: desativa neurônios INDIVIDUAIS aleatoriamente
    # Bom para: camadas Dense (FC) onde cada neurônio é independente
    # Dropout(0.5) ← 50% dos neurônios desativados por batch

    # SpatialDropout2D: desativa FEATURE MAPS INTEIROS aleatoriamente
    # Bom para: camadas Conv onde pixels vizinhos são correlacionados
    # Se pixels adjacentes são correlatos, dropout por pixel não é tao eficiente
    # SpatialDropout2D(0.2) ← 20% dos canais (feature maps) completos desativados
    # Dropout(0.5)          # Dropout padrão: desativa neurônios individualmente
    # SpatialDropout2D(0.2) # Desativa feature maps inteiros

    # ─── VISUALIZAÇÃO: DROPOUT vs SPATIALDROPOUT2D ───
    np.random.seed(5)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    n_neurons = 8
    fm_h, fm_w = 4, 4
    n_fm = 6  # feature maps

    # Dropout(0.5): desativa 50% dos neurônios aleatoriamente
    dropout_mask = np.random.rand(n_neurons) > 0.5
    ax = axes[0]
    ax.set_title('Dropout(0.5)\n(Dense layer: por neurônio)', fontsize=11, fontweight='bold')
    for i, active in enumerate(dropout_mask):
        y = n_neurons - i
        color = '#4e79a7' if active else '#e15759'
        status = 'Ativo' if active else 'Desativado'
        rect = mpatches.FancyBboxPatch((0.5, y - 0.35), 2, 0.65,
                                        boxstyle='round,pad=0.05', facecolor=color, edgecolor='black',
                                        alpha=0.8 if active else 0.3)
        ax.add_patch(rect)
        ax.text(1.5, y, f'N{i+1}: {status}', ha='center', va='center', fontsize=9,
                color='white' if active else 'gray')
    ax.set_xlim(0, 4)
    ax.set_ylim(0, n_neurons + 1)
    ax.axis('off')
    n_active = dropout_mask.sum()
    ax.text(1.5, 0.3, f'{n_active}/{n_neurons} ativos ({n_active/n_neurons*100:.0f}%)',
            ha='center', fontsize=10, color='navy', fontweight='bold')

    # SpatialDropout2D(0.2): desativa 20% dos feature maps completos
    spatial_mask = np.random.rand(n_fm) > 0.2
    ax2 = axes[1]
    ax2.set_title('SpatialDropout2D(0.2)\n(Conv layer: por feature map)', fontsize=11, fontweight='bold')
    for fm_idx, active in enumerate(spatial_mask):
        col = fm_idx % 3
        row = fm_idx // 3
        x0 = col * 1.8 + 0.2
        y0 = (1 - row) * 2.5 + 0.5
        data = np.random.rand(fm_h, fm_w) if active else np.zeros((fm_h, fm_w))
        sub = ax2.inset_axes([x0/5.5, y0/3.5, 1.4/5.5, 1.8/3.5])
        sub.imshow(data, cmap='Blues', vmin=0, vmax=1)
        sub.set_title(f'FM{fm_idx+1}' + ('' if active else ' ✗'), fontsize=8,
                       color='black' if active else 'red')
        sub.axis('off')
    ax2.set_xlim(0, 6)
    ax2.set_ylim(0, 4)
    ax2.axis('off')
    n_a = spatial_mask.sum()
    ax2.text(2.7, 0.1, f'{n_a}/{n_fm} feature maps ativos', ha='center', fontsize=10,
             color='navy', fontweight='bold')

    # Comparação: quando usar cada um
    ax3 = axes[2]
    categorias = ['Dropout\n(0.5)\nDense', 'SpatialDropout2D\n(0.2)\nConv']
    pros = ['Por neurônio\nPreserva topologia\nBom para FC layers',
            'Por feature map\nRegulariza espacialmente\nBom para CNNs']
    colors_bar = ['#4e79a7', '#e15759']
    bars = ax3.barh(categorias, [0.65, 0.35], color=colors_bar, edgecolor='black', alpha=0.8)
    ax3.set_xlim(0, 1.0)
    ax3.set_title('Quando Usar Cada Tipo\nde Dropout', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Aplicabilidade relativa em CNNs')
    for bar, pro in zip(bars, pros):
        ax3.text(0.02, bar.get_y() + bar.get_height()/2, pro, va='center', fontsize=8, color='white')
    ax3.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Dropout vs SpatialDropout2D — Comparação Visual',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
