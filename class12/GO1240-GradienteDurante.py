# GO1240-GradienteDurante
# Gradiente durante backprop com skip connection:
#
#   ∂Loss/∂x = ∂Loss/∂H × ∂H/∂x
#            = ∂Loss/∂H × (∂F/∂x + 1)   ← "+1" vem da skip connection!
#
# Skip connection = "autoestrada" para o gradiente:
# Não passa por ativações → não sofre saturação → sem vanishing gradient!
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass

# ─── VISUALIZAÇÃO: FLUXO DO GRADIENTE COM E SEM SKIP CONNECTION ───
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

def draw_network_gradient(ax, title, with_skip):
    ax.set_xlim(-0.5, 5)
    ax.set_ylim(-0.5, 7)
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold')

    camadas = ['x\n(entrada)', 'Layer 1\nReLU', 'Layer 2\nReLU', 'Layer 3\nReLU', 'Loss']
    y_pos = [1, 2.5, 4, 5.5, 7]
    colors_fwd = ['#59a14f', '#4e79a7', '#4e79a7', '#4e79a7', '#e15759']

    # Forward pass (setas para cima)
    for i, (nome, y, cor) in enumerate(zip(camadas, y_pos, colors_fwd)):
        rect = mpatches.FancyBboxPatch((1.5, y - 0.35), 2, 0.7,
                                        boxstyle='round,pad=0.05', linewidth=1.5,
                                        edgecolor='black', facecolor=cor, alpha=0.8)
        ax.add_patch(rect)
        ax.text(2.5, y, nome, ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        if i < len(camadas) - 1:
            ax.annotate('', xy=(2.5, y_pos[i+1] - 0.35), xytext=(2.5, y + 0.35),
                        arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))

    # Simular magnitudes do gradiente em cada camada:
    # Sem skip: cada ReLU multiplica por ~0.5 (só pass a gradiente para neurônios ativos)
    #   ∂L/∂x = 1.0 × 0.5 × 0.5 × 0.5 ≈ 0.1 (vanishing gradient!)
    # Com skip: o "+1" garante que o gradiente nunca fica menor que 1.0
    #   ∂L/∂x = 1.0 → 0.95 → 0.90 → 0.85 (quase sem perda!)
    grad_size = [1.0, 0.5, 0.25, 0.1] if not with_skip else [1.0, 0.95, 0.9, 0.85]
    grad_colors = ['#e15759'] * 4
    for i in range(len(y_pos) - 1):
        g = grad_size[i]
        ax.annotate('', xy=(1.2, y_pos[i] + 0.35), xytext=(1.2, y_pos[i+1] - 0.35),
                    arrowprops=dict(arrowstyle='->', color='#e15759', lw=max(0.5, g*3)))
        ax.text(0.7, (y_pos[i] + y_pos[i+1]) / 2, f'∇≈{g:.2f}',
                ha='center', va='center', fontsize=8, color='#e15759', fontweight='bold')

    # Skip connection
    if with_skip:
        ax.annotate('', xy=(3.8, y_pos[0] + 0.35), xytext=(3.8, y_pos[2] - 0.35),
                    arrowprops=dict(arrowstyle='->', color='#59a14f', lw=2.5,
                                    connectionstyle='arc3,rad=0.3'))
        ax.text(4.4, (y_pos[0] + y_pos[2]) / 2, 'Skip\n(+1)', ha='center', va='center',
                fontsize=9, color='#59a14f', fontweight='bold')
        ax.text(2.5, 0.2, '✅ Gradiente preservado!', ha='center', fontsize=9,
                color='#59a14f', fontweight='bold')
    else:
        ax.text(2.5, 0.2, '⚠️ Vanishing Gradient!', ha='center', fontsize=9,
                color='#e15759', fontweight='bold')

draw_network_gradient(axes[0], 'Sem Skip Connection\n(Vanishing Gradient)', with_skip=False)
draw_network_gradient(axes[1], 'Com Skip Connection (ResNet)\n(Gradiente preservado)', with_skip=True)

plt.suptitle('Skip Connections — Autoestrada para o Gradiente no Backprop',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# ─── VISUALIZAÇÃO: GRADIENTE ACUMULADO POR PROFUNDIDADE ───
np.random.seed(3)
depths = np.arange(1, 21)
grad_plain = 1.0 * (0.7 ** depths) + 0.01 * np.random.randn(20)
grad_resnet = 1.0 * np.ones(20) - depths * 0.008 + 0.01 * np.random.randn(20)
grad_resnet = np.clip(grad_resnet, 0.3, 1.0)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(depths, grad_plain,  'o-', color='#e15759', linewidth=2, label='Rede Profunda (sem skip)')
ax.plot(depths, grad_resnet, 's-', color='#4e79a7', linewidth=2, label='ResNet (com skip connections)')
ax.fill_between(depths, grad_plain, 0.05, alpha=0.1, color='red')
ax.axhline(0.05, color='red', linestyle=':', linewidth=1, label='Limiar de vanishing gradient')
ax.set_xlabel('Profundidade da Camada (distância da saída)')
ax.set_ylabel('Magnitude do Gradiente')
ax.set_title('Magnitude do Gradiente por Profundidade', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
