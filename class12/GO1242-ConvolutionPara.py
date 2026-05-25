# GO1242-ConvolutionPara
# Convolucão 1×1 — "Network-in-Network" (Lin et al., 2013)
# Objetivo: ajustar o número de canais (dimensão de profundidade) SEM alterar H×W
# Usos:
#   1. Manter dimenso (identity skip quando in_ch == out_ch)
#   2. Aumentar canais (projetar para espaço de maior dimensão)
#   3. Comprimir canais (bottleneck: reduz computacão antes de conv 3×3)
#      Ex: ResNet bottleneck: 256 → 64 (1×1) → 64 (3×3) → 256 (1×1)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass

# ─── VISUALIZAÇÃO: CONVOLUIÇÃO 1×1 ───
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

def draw_tensor(ax, x, y, w, h, d, color, label, alpha=0.7):
    """Desenha um tensor 3D como retângulo com profundidade"""
    # Face frontal
    rect = plt.Polygon([[x, y], [x+w, y], [x+w, y+h], [x, y+h]],
                         closed=True, facecolor=color, edgecolor='black', alpha=alpha, linewidth=1.5)
    ax.add_patch(rect)
    # Face superior
    top = plt.Polygon([[x, y+h], [x+w, y+h], [x+w+d, y+h+d], [x+d, y+h+d]],
                       closed=True, facecolor=color, edgecolor='black', alpha=alpha*0.7, linewidth=1.5)
    ax.add_patch(top)
    # Face lateral
    side = plt.Polygon([[x+w, y], [x+w+d, y+d], [x+w+d, y+h+d], [x+w, y+h]],
                        closed=True, facecolor=color, edgecolor='black', alpha=alpha*0.85, linewidth=1.5)
    ax.add_patch(side)
    ax.text(x + w/2, y - 0.3, label, ha='center', fontsize=9, fontweight='bold')

for ax_i, (in_ch, out_ch, title) in enumerate([
    (64, 64,  '1×1 Conv: mesma dimensão\n(passa sem mudar canais)'),
    (64, 128, '1×1 Conv: aumentar canais\n(64 → 128 canais)'),
    (256, 64, '1×1 Conv: comprimir canais\n(Bottleneck: 256 → 64)'),
]):
    ax = axes[ax_i]
    ax.set_xlim(-0.5, 7)
    ax.set_ylim(-0.8, 4)
    ax.axis('off')
    ax.set_title(title, fontsize=10, fontweight='bold')

    w_in  = min(2.0, in_ch / 64)
    w_out = min(2.0, out_ch / 64)
    h = 2
    # Tensor de entrada
    draw_tensor(ax, 0, 0.5, 1.2, h, w_in * 0.5, '#4e79a7', f'Input\n7×7×{in_ch}')
    # Kernel 1×1
    draw_tensor(ax, 2.2, 0.8, 0.3, 0.5, w_in * 0.5 + 0.1, '#e15759', f'1×1\n×{out_ch}', alpha=0.9)
    ax.annotate('', xy=(2.2, 1.8), xytext=(1.2 + w_in*0.5, 1.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    # Tensor de saída
    draw_tensor(ax, 3.3, 0.5, 1.2, h, w_out * 0.5, '#59a14f', f'Output\n7×7×{out_ch}')
    ax.annotate('', xy=(3.3, 1.8), xytext=(2.5 + 0.3, 1.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Informações
    params = in_ch * out_ch  # 1×1 kernel: in_ch * out_ch parâmetros
    ax.text(3.5, -0.5, f'Parâmetros: {params:,}\n(apenas {in_ch}×{out_ch}, sem overhead espacial)',
            ha='left', fontsize=8, color='navy')

plt.suptitle('Convoluição 1×1 — Ajuste de Dimensões nos Blocos Residuais',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
