# GO1258-Codigo
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass


if __name__ == '__main__':
    from tensorflow.keras.callbacks import EarlyStopping

    es = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # ─── VISUALIZAÇÃO: EARLYSTOPPING COM restore_best_weights ───
    np.random.seed(12)
    n_epochs = 45
    epochs = np.arange(1, n_epochs + 1)

    # Simular: val_loss melhora até época 22, depois piora
    best_ep = 22
    val_loss = np.concatenate([
        2.0 * np.exp(-0.22 * np.arange(1, best_ep + 1)),
        0.18 + 0.015 * np.arange(0, n_epochs - best_ep)
    ]) + 0.015 * np.random.randn(n_epochs)
    train_loss = 2.0 * np.exp(-0.28 * epochs) + 0.05 + 0.01 * np.random.randn(n_epochs)

    stop_ep = best_ep + 5   # patience=5

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(epochs[:stop_ep], train_loss[:stop_ep], 'o-', color='#4e79a7', linewidth=2,
            markersize=3, label='Train Loss')
    ax.plot(epochs[:stop_ep], val_loss[:stop_ep],   's-', color='#e15759', linewidth=2,
            markersize=3, label='Val Loss')

    # Marcações
    ax.axvline(best_ep, color='green', linestyle='--', linewidth=2.5,
               label=f'Melhor Modelo salvo (ép.{best_ep})')
    ax.axvline(stop_ep, color='red',   linestyle=':',  linewidth=2.5,
               label=f'EarlyStopping (ép.{stop_ep})')
    ax.plot(best_ep, val_loss[best_ep-1], 'D', color='green', markersize=12, zorder=5)
    ax.fill_between(epochs[best_ep-1:stop_ep], 0, 2,
                     alpha=0.10, color='red', label=f'Sem melhora (patience=5)')
    ax.annotate('Pesos restaurados para\neste ponto de mínimo',
                xy=(best_ep, val_loss[best_ep-1]),
                xytext=(best_ep + 4, val_loss[best_ep-1] + 0.3),
                fontsize=10, color='green', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=1.8))

    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('EarlyStopping(patience=5, restore_best_weights=True)\n'
                 'Treino para quando val_loss não melhora por 5 épocas consecutivas',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 2.1)
    plt.tight_layout()
    plt.show()
