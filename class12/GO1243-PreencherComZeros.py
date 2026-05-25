# GO1243-PreencherComZeros
# Preencher com zeros nos canais extras (zero-padding de canais)
#   identity = F.pad(x, [0, 0, 0, 0, 0, out_channels - in_channels])
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass

# ─── VISUALIZAÇÃO: ZERO-PADDING DE CANAIS vs PROJEÇÃO 1×1 ───
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

np.random.seed(1)
in_ch  = 4
out_ch = 8

# Simular canais de entrada
canais_entrada = np.random.rand(in_ch, 5, 5) * 2 - 1

# Zero-padding: primeiros in_ch = dados, resto = zeros
zero_padded = np.zeros((out_ch, 5, 5))
zero_padded[:in_ch] = canais_entrada

# Projeção 1×1: todos os canais têm informação (simular)
proj_1x1 = np.random.rand(out_ch, 5, 5) * 1.2 - 0.6

# Plotar canais como heatmaps
for ax, data, title, subtitle in [
    (axes[0], zero_padded, 'Zero-Padding de Canais\n(F.pad)', f'{in_ch} canais reais + {out_ch - in_ch} zeros'),
    (axes[1], proj_1x1,   'Projeção 1×1\n(Conv2D kernel 1×1)',  f'Todos os {out_ch} canais com informação'),
]:
    n_cols = 4
    n_rows = out_ch // n_cols
    inner = ax.get_position()
    ax.axis('off')
    ax.set_title(title + f'\n{subtitle}', fontsize=11, fontweight='bold')
    for idx in range(out_ch):
        sub_ax = fig.add_axes([
            inner.x0 + (idx % n_cols) * inner.width / n_cols + 0.005,
            inner.y0 + (n_rows - 1 - idx // n_cols) * inner.height / n_rows + 0.04,
            inner.width / n_cols - 0.01,
            inner.height / n_rows - 0.05
        ])
        is_zero = (ax == axes[0] and idx >= in_ch)
        sub_ax.imshow(data[idx], cmap='RdBu', vmin=-1, vmax=1)
        sub_ax.set_title(f'C{idx+1}' + (' (zero)' if is_zero else ''),
                         fontsize=7, color='red' if is_zero else 'black')
        sub_ax.set_xticks([])
        sub_ax.set_yticks([])

plt.suptitle('Estratégias para Ajustar Dimensão da Skip Connection (Blocos Residuais)',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
