# GO1253-Callbacks
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass


if __name__ == '__main__':
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    callbacks = [
        # EarlyStopping: para o treino quando val_loss não melhora
        # patience=5: aguarda 5 épocas sem melhora antes de parar
        # restore_best_weights=True (padrão=False): restaura pesos da melhor época
        EarlyStopping(monitor='val_loss', patience=5),
        # ModelCheckpoint: salva o modelo quando val_loss melhora
        # save_best_only=True: não sobrescreve se não houver melhora
        ModelCheckpoint('best.h5', save_best_only=True),
        # ReduceLROnPlateau: reduz LR por `factor` quando val_loss estagna
        # factor=0.5: LR novo = LR atual × 0.5 (divide por 2)
        # patience=3: aguarda 3 épocas antes de reduzir
        ReduceLROnPlateau(factor=0.5, patience=3)
    ]

    # ─── VISUALIZAÇÃO: EFEITO DOS CALLBACKS NO TREINAMENTO ───
    np.random.seed(8)
    n_epochs = 35
    epochs = np.arange(1, n_epochs + 1)

    # Simular métricas
    val_loss   = np.concatenate([2.0 * np.exp(-0.2 * np.arange(1, 16)),
                                  0.35 + 0.01 * np.arange(0, 12),      # estagna e piora
                                  0.35 + 0.008 * np.arange(0, 8)]) + 0.02 * np.random.randn(35)
    train_loss = 2.0 * np.exp(-0.22 * epochs) + 0.05 + 0.01 * np.random.randn(35)
    lr = np.ones(35) * 0.001
    lr[18:] = 0.0005   # ReduceLROnPlateau na época 18
    lr[25:] = 0.00025  # segunda redução
    best_epoch = 15    # melhor val_loss
    stop_epoch = 20    # EarlyStopping (patience=5)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Loss com EarlyStopping
    axes[0].plot(epochs[:stop_epoch], train_loss[:stop_epoch], 'o-', color='#4e79a7',
                 linewidth=2, markersize=3, label='Train Loss')
    axes[0].plot(epochs[:stop_epoch], val_loss[:stop_epoch],   's-', color='#e15759',
                 linewidth=2, markersize=3, label='Val Loss')
    axes[0].axvline(best_epoch, color='green', linestyle='--', linewidth=2, label=f'Best model (ép.{best_epoch})')
    axes[0].axvline(stop_epoch, color='red',   linestyle=':',  linewidth=2, label=f'EarlyStopping (ép.{stop_epoch})')
    axes[0].fill_between(epochs[best_epoch-1:stop_epoch], 0, 2,
                          alpha=0.08, color='red', label='Sem melhora (patience=5)')
    axes[0].set_title('EarlyStopping\n(para quando val_loss não melhora por 5 épocas)', fontsize=10)
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 2.1)

    # ModelCheckpoint
    checkpoint_epochs = [e for e in range(1, stop_epoch + 1) if val_loss[e-1] == min(val_loss[:e])]
    axes[1].plot(epochs[:stop_epoch], val_loss[:stop_epoch], 's-', color='#e15759',
                 linewidth=2, markersize=3, label='Val Loss')
    for ep in checkpoint_epochs:
        axes[1].axvline(ep, color='green', linestyle=':', linewidth=1.2, alpha=0.8)
        axes[1].plot(ep, val_loss[ep-1], 'D', color='green', markersize=10, zorder=5)
    axes[1].plot([], [], 'D-', color='green', markersize=8, label='Modelo salvo (best.h5)')
    axes[1].set_title('ModelCheckpoint\n(salva sempre que val_loss melhora)', fontsize=10)
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Val Loss')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 2.1)

    # ReduceLROnPlateau
    ax_lr = axes[2].twinx()
    axes[2].plot(epochs[:stop_epoch], val_loss[:stop_epoch], 's-', color='#e15759',
                 linewidth=2, markersize=3, label='Val Loss')
    ax_lr.plot(epochs[:stop_epoch], lr[:stop_epoch] * 1000, '^--', color='#4e79a7',
               linewidth=2, markersize=5, label='LR (×1000)')
    axes[2].axvline(18, color='navy', linestyle=':', linewidth=1.5, label='LR ÷ 2 (ép.18)')
    axes[2].axvline(25, color='navy', linestyle=':', linewidth=1.5)
    axes[2].set_title('ReduceLROnPlateau\n(divide LR por 2 após 3 épocas sem melhora)', fontsize=10)
    axes[2].set_xlabel('Época')
    axes[2].set_ylabel('Val Loss', color='#e15759')
    ax_lr.set_ylabel('Learning Rate (×1000)', color='#4e79a7')
    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax_lr.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Callbacks — EarlyStopping + ModelCheckpoint + ReduceLROnPlateau',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
