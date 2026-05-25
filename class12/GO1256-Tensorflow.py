# GO1256-Tensorflow
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass


if __name__ == '__main__':
    # Conv2D(64, (3,3), kernel_regularizer=l2(0.01))
    # Dense(128, kernel_regularizer=l2(0.01))

    # ─── VISUALIZAÇÃO: EFEITO DA REGULARIZAÇÃO L2 ───
    np.random.seed(42)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Distribuição dos pesos
    sem_l2 = np.random.randn(1000) * 2      # pesos grandes, sem penalização
    com_l2 = np.random.randn(1000) * 0.5   # pesos menores, regularizados

    axes[0].hist(sem_l2, bins=50, color='#e15759', alpha=0.7, density=True, label=f'Sem L2: std={sem_l2.std():.2f}')
    axes[0].hist(com_l2, bins=50, color='#4e79a7', alpha=0.7, density=True, label=f'Com L2: std={com_l2.std():.2f}')
    axes[0].set_title('Distribuição dos Pesos\nSem vs Com L2 (\u03bb=0.01)', fontsize=11)
    axes[0].set_xlabel('Valor do Peso')
    axes[0].set_ylabel('Densidade')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss curves
    epochs = np.arange(1, 51)
    train_sem = 2.0 * np.exp(-0.25 * epochs) + 0.05 + 0.01 * np.random.randn(50)
    val_sem   = 2.0 * np.exp(-0.15 * epochs) + 0.6  + 0.02 * np.random.randn(50)  # overfitting
    train_com = 2.0 * np.exp(-0.22 * epochs) + 0.15 + 0.01 * np.random.randn(50)
    val_com   = 2.0 * np.exp(-0.20 * epochs) + 0.2  + 0.01 * np.random.randn(50)  # generaliza bem

    for ax, tr, va, title in [
        (axes[1], train_sem, val_sem, 'Sem L2 (overfitting)'),
        (axes[2], train_com, val_com, 'Com L2=0.01 (regularizado)'),
    ]:
        ax.plot(epochs, tr, '-', color='#4e79a7', linewidth=2, label='Train Loss')
        ax.plot(epochs, va, '-', color='#e15759', linewidth=2, label='Val Loss')
        gap = (va[-10:] - tr[-10:]).mean()
        ax.fill_between(epochs, tr, va, alpha=0.1, color='red', label=f'Gap médio: {gap:.3f}')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Época')
        ax.set_ylabel('Loss')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 2.2)

    plt.suptitle('Regularização L2 — Impacto nos Pesos e na Generalização',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
