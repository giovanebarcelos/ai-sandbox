# GO1255-ReduzirLrQuando
# Reduzir LR quando val_loss estagnar
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass


if __name__ == "__main__":
    from tensorflow.keras.callbacks import ReduceLROnPlateau
    from tensorflow.keras.optimizers.schedules import ExponentialDecay

    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,        # Multiplicar LR por 0.5
        patience=3,        # Aguardar 3 épocas
        min_lr=1e-7
    )

    # Decay exponencial
    ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.9
    )

    # ─── VISUALIZAÇÃO: CURVAS DE LEARNING RATE ───
    epochs = np.arange(1, 51)

    # Simular ReduceLROnPlateau: LR cai 50% a cada 3 épocas de estagnação
    lr_plateau = np.ones(50) * 0.001
    reductions = [10, 14, 18, 25, 30, 38]  # Épocas onde o LR é reduzido
    current_lr = 0.001
    for i in range(50):
        if i in reductions:
            current_lr *= 0.5
        lr_plateau[i] = max(current_lr, 1e-7)

    # Exponential Decay
    initial_lr = 0.001
    decay_rate = 0.9
    decay_steps = 5  # cada 5 épocas
    lr_exp = initial_lr * (decay_rate ** (epochs / decay_steps))

    # Cosine Annealing
    lr_cosine = 1e-6 + 0.5 * (0.001 - 1e-6) * (1 + np.cos(np.pi * epochs / 50))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Comparação de schedulers
    ax1.plot(epochs, lr_plateau, 'o-', color='#4e79a7', linewidth=2, markersize=3,
             label='ReduceLROnPlateau\n(fator=0.5, patience=3)')
    ax1.plot(epochs, lr_exp, 's-', color='#f28e2b', linewidth=2, markersize=3,
             label='ExponentialDecay\n(rate=0.9, steps=5)')
    ax1.plot(epochs, lr_cosine, '^-', color='#e15759', linewidth=2, markersize=3,
             label='Cosine Annealing')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Learning Rate')
    ax1.set_title('Comparação: Estratégias de LR Scheduling', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Impacto do LR na convergência (simulado)
    np.random.seed(7)
    loss_fixo = 2.5 * np.exp(-0.06 * epochs) + 0.5 + 0.03 * np.random.randn(50)
    loss_schedule = 2.5 * np.exp(-0.1 * epochs) + 0.15 + 0.02 * np.random.randn(50)
    loss_schedule[20:] = loss_schedule[20:] * 0.8  # Melhora após redução

    ax2.plot(epochs, loss_fixo, linewidth=2, color='#e15759', label='LR Fixo (0.001)')
    ax2.plot(epochs, loss_schedule, linewidth=2, color='#4e79a7', label='LR Scheduling\n(ReduceLROnPlateau)')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Val Loss')
    ax2.set_title('Impacto do LR Scheduling na Convergência', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.suptitle('Learning Rate Scheduling', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
