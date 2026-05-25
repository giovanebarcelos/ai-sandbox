# GO1241-ResidualBlock
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Bloco residual básico (usado em ResNet-18/34)
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Caminho principal (F(x))
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection (identity)
        self.shortcut = nn.Sequential()

        # Se dimensões não batem, usar projeção 1×1
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Salvar input para skip connection
        identity = x

        # Caminho principal: F(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection: F(x) + x
        identity = self.shortcut(identity)  # Ajustar dimensões se necessário
        out += identity  # ⭐ ELEMENTO-WISE ADD!

        # ReLU DEPOIS da adição
        out = self.relu(out)

        return out


if __name__ == '__main__':
    import torch

    print("=== Demonstração do Bloco Residual ===")

    # Caso 1: mesma dimensão (identity shortcut)
    block_id = ResidualBlock(in_channels=64, out_channels=64)
    x = torch.randn(1, 64, 56, 56)
    out = block_id(x)
    print(f"Identity shortcut:")
    print(f"  Entrada:  {x.shape}")
    print(f"  Saída:    {out.shape}")
    print(f"  Erro L2 (residual vs entrada): {(out - x).norm().item():.4f}")

    # Caso 2: mudança de dimensão (projection shortcut)
    block_proj = ResidualBlock(in_channels=64, out_channels=128, stride=2)
    out_proj = block_proj(x)
    print(f"\nProjection shortcut (stride=2, 64→128ch):")
    print(f"  Entrada:  {x.shape}")
    print(f"  Saída:    {out_proj.shape}")

    # Contar parâmetros
    params = sum(p.numel() for p in block_proj.parameters())
    print(f"  Parâmetros: {params:,}")

    # ─── VISUALIZAÇÃO: DIAGRAMA DO BLOCO RESIDUAL ───
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch
    import numpy as np

    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
    except NameError:
        pass

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Diagrama do bloco residual ---
    ax = axes[0]
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Bloco Residual (ResNet)', fontsize=13, fontweight='bold')

    def draw_box(ax, x, y, w, h, text, color='#4e79a7', fontsize=9):
        rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                        boxstyle='round,pad=0.1', linewidth=1.5,
                                        edgecolor='black', facecolor=color, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color='white')

    # Caminho principal
    draw_box(ax, 4, 9, 2.5, 0.7, 'x  (entrada)', color='#59a14f')
    draw_box(ax, 4, 7.5, 2.5, 0.7, 'Conv 3×3 + BN', color='#4e79a7')
    draw_box(ax, 4, 6.2, 2.5, 0.7, 'ReLU', color='#f28e2b')
    draw_box(ax, 4, 4.9, 2.5, 0.7, 'Conv 3×3 + BN', color='#4e79a7')
    draw_box(ax, 4, 3.4, 2.5, 0.9, '⊕  Add (F(x) + x)', color='#e15759')
    draw_box(ax, 4, 2.1, 2.5, 0.7, 'ReLU', color='#f28e2b')
    draw_box(ax, 4, 0.8, 2.5, 0.7, 'Saída', color='#59a14f')

    # Setas do caminho principal
    for y1, y2 in [(8.65, 7.85), (7.15, 6.55), (5.85, 5.25), (4.55, 3.85), (2.95, 2.45), (1.75, 1.15)]:
        ax.annotate('', xy=(4, y2), xytext=(4, y1),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Skip connection
    ax.annotate('', xy=(6, 3.4), xytext=(6, 9),
                arrowprops=dict(arrowstyle='->', color='#e15759', lw=2.5,
                                connectionstyle='arc3,rad=-0.3'))
    ax.text(7.0, 6.2, 'Skip\nConnection\n(identidade)', ha='center', va='center',
            fontsize=9, color='#e15759', fontweight='bold')

    # --- Comparação: sem vs com skip connections ---
    ax2 = axes[1]
    np.random.seed(42)
    epochs = np.arange(1, 21)
    # Simular curvas de perda - rede profunda sem skip (vanishing gradient)
    loss_sem = 2.5 * np.exp(-0.08 * epochs) + 0.8 + 0.05 * np.random.randn(20)
    loss_com = 2.5 * np.exp(-0.22 * epochs) + 0.2 + 0.02 * np.random.randn(20)

    ax2.plot(epochs, loss_sem, 'o-', color='#e15759', linewidth=2, label='Sem Skip Connections\n(degradação de gradiente)', markersize=5)
    ax2.plot(epochs, loss_com, 's-', color='#4e79a7', linewidth=2, label='Com Skip Connections\n(ResNet - treino eficaz)', markersize=5)
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Loss')
    ax2.set_title('Impacto das Skip Connections no Treinamento', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.fill_between(epochs, loss_sem, loss_com, alpha=0.1, color='green',
                     label='Melhoria')
    ax2.text(12, 1.2, '↑ Melhoria com\nskip connections', fontsize=9,
             color='green', fontweight='bold')

    plt.suptitle('ResNet - Blocos Residuais e Skip Connections', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
