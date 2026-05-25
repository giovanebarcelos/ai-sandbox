# GO1233-Durante
# Durante TREINO: Dropout desativa neurônios aleatoriamente por batch
# Durante TESTE:  Usa todos os neurônios (pesos escalonados por 1-p)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass


if __name__ == '__main__':
    # Conceito: Dropout train vs test
    np.random.seed(42)
    n_neurons = 8
    dropout_rate = 0.5

    # Máscara de dropout (50%)
    mask = np.random.rand(n_neurons) > dropout_rate

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def draw_layer(ax, title, active_mask, scale=1.0):
        ax.set_xlim(-1, 3)
        ax.set_ylim(-0.5, n_neurons + 0.5)
        ax.axis('off')
        ax.set_title(title, fontsize=12, fontweight='bold')
        for i in range(n_neurons):
            color = '#4e79a7' if active_mask[i] else '#dddddd'
            lw = 2 if active_mask[i] else 0.5
            circle = plt.Circle((1, i), 0.38, color=color, linewidth=lw, ec='black')
            ax.add_patch(circle)
            if active_mask[i]:
                ax.text(1, i, f'×{scale:.1f}', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
            else:
                ax.text(1, i, '✕', ha='center', va='center', fontsize=10, color='#aaaaaa')
            # Setas de entrada
            ax.annotate('', xy=(0.62, i), xytext=(-0.5, i),
                        arrowprops=dict(arrowstyle='->', color=color, lw=lw))
            # Setas de saída
            ax.annotate('', xy=(2.5, i), xytext=(1.38, i),
                        arrowprops=dict(arrowstyle='->', color=color, lw=lw))

    draw_layer(axes[0], f'TREINO — Batch 1\n(dropout=50%, {mask.sum()}/8 ativos)', mask)
    draw_layer(axes[1], f'TREINO — Batch 2\n(nova máscara aleatória)', np.random.rand(n_neurons) > dropout_rate)
    draw_layer(axes[2], 'TESTE/INFERÊNCIA\n(todos ativos, pesos ÷ (1-p))', np.ones(n_neurons, dtype=bool), scale=0.5)

    # Legenda
    axes[0].add_patch(mpatches.Patch(color='#4e79a7', label='Neurônio ativo'))
    axes[0].add_patch(mpatches.Patch(color='#dddddd', label='Neurônio desativado'))
    axes[0].legend(loc='lower left', fontsize=9)

    plt.suptitle('Dropout — Treino vs Teste\n"Cada batch treina uma sub-rede diferente"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Comparação: com vs sem Dropout
    epochs = np.arange(1, 31)
    np.random.seed(5)
    train_no_drop = 0.9 - 0.8 * np.exp(-0.2 * epochs) + 0.01 * np.random.randn(30)
    val_no_drop   = 0.65 - 0.5 * np.exp(-0.15 * epochs) + 0.02 * np.random.randn(30)
    train_drop    = 0.85 - 0.75 * np.exp(-0.18 * epochs) + 0.01 * np.random.randn(30)
    val_drop      = 0.82 - 0.72 * np.exp(-0.17 * epochs) + 0.01 * np.random.randn(30)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_no_drop, '--', color='#e15759', label='Train (sem Dropout)')
    ax.plot(epochs, val_no_drop,   '-',  color='#e15759', label='Val   (sem Dropout) — overfitting!')
    ax.plot(epochs, train_drop,    '--', color='#4e79a7', label='Train (com Dropout)')
    ax.plot(epochs, val_drop,      '-',  color='#4e79a7', label='Val   (com Dropout) — melhor generalização!')
    ax.fill_between(epochs, train_no_drop, val_no_drop, alpha=0.1, color='#e15759', label='Gap (overfitting)')
    ax.fill_between(epochs, train_drop,    val_drop,    alpha=0.1, color='#4e79a7')
    ax.set_xlabel('Época')
    ax.set_ylabel('Accuracy')
    ax.set_title('Dropout — Redução do Overfitting', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
