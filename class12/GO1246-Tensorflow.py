# GO1246-Tensorflow
from tensorflow.keras.layers import Dropout
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass

# model.add(Dropout(0.5))

# ─── VISUALIZAÇÃO: IMPACTO DA TAXA DE DROPOUT NA CAPACIDADE E GENERALIZAÇÃO ───
np.random.seed(42)
epochs = np.arange(1, 41)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Simular train/val loss para diferentes taxas de dropout
rates = [0.0, 0.2, 0.5, 0.7]
colors = ['#e15759', '#f28e2b', '#4e79a7', '#76b7b2']

for rate, color in zip(rates, colors):
    # Train loss: maior dropout → mais ruído no treino
    noise = 0.03 + rate * 0.05
    train_loss = 2.5 * np.exp(-(0.2 - rate * 0.1) * epochs) + 0.1 + rate * 0.2 + noise * np.random.randn(40)
    # Val loss: moderado dropout melhora generalização
    if rate == 0.0:
        val_loss = 2.5 * np.exp(-0.15 * epochs) + 0.5 + 0.03 * np.random.randn(40)  # overfitting
    elif rate == 0.5:
        val_loss = 2.5 * np.exp(-0.18 * epochs) + 0.15 + 0.02 * np.random.randn(40)  # melhor
    else:
        coeff = 0.3 + abs(rate - 0.5) * 1.5
        val_loss = 2.5 * np.exp(-0.16 * epochs) + coeff * 0.4 + 0.02 * np.random.randn(40)

    axes[0].plot(epochs, train_loss, '--', color=color, linewidth=1.5, alpha=0.7)
    axes[1].plot(epochs, val_loss,   '-',  color=color, linewidth=2, label=f'Dropout={rate}')

axes[0].set_title('Loss de Treino por Taxa de Dropout\n(tracejado)', fontsize=11)
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Train Loss')
axes[0].grid(True, alpha=0.3)
for rate, color, lbl in zip(rates, colors, [f'={r}' for r in rates]):
    axes[0].plot([], [], '--', color=color, label=f'Dropout{lbl}')
axes[0].legend(fontsize=9)

axes[1].set_title('Loss de Validação por Taxa de Dropout\n(ótimo: 0.5)', fontsize=11)
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Val Loss')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
# Destacar o dropout=0.5
axes[1].annotate('← Melhor generalização\n  Dropout=0.5', xy=(25, 0.4), fontsize=9,
                 color='#4e79a7', fontweight='bold')

plt.suptitle('Dropout — Impacto da Taxa na Generalização', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
