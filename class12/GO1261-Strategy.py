# GO1261-Strategy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass

# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():
#     model = create_model()  # Distribui automaticamente

# ─── VISUALIZAÇÃO: MIRRORED STRATEGY (MULTI-GPU) ───
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Diagrama da arquitetura
ax = axes[0]
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('MirroredStrategy — Treinamento Multi-GPU\n(copia idêntica do modelo em cada GPU)', fontsize=11, fontweight='bold')

# Modelo mestre
center_rect = mpatches.FancyBboxPatch((3.5, 8.2), 5, 1.2,
                                       boxstyle='round,pad=0.1', facecolor='#4e79a7',
                                       edgecolor='black', linewidth=2, alpha=0.9)
ax.add_patch(center_rect)
ax.text(6, 8.85, 'Modelo Espelhado\n(pesos idênticos)', ha='center', va='center',
         fontsize=10, fontweight='bold', color='white')

# GPUs
gpu_colors = ['#e15759', '#59a14f', '#f28e2b', '#76b7b2']
n_gpus = 4
for i in range(n_gpus):
    x = 1 + i * 2.7
    y = 5.0
    # GPU box
    rect = mpatches.FancyBboxPatch((x, y), 2.2, 1.5,
                                    boxstyle='round,pad=0.1', facecolor=gpu_colors[i],
                                    edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.add_patch(rect)
    ax.text(x + 1.1, y + 0.75, f'GPU {i}\n(replica)', ha='center', va='center',
             fontsize=9, fontweight='bold', color='white')
    # Seta de cima para GPU
    ax.annotate('', xy=(x + 1.1, y + 1.5), xytext=(4.5 + i * 0.7, 8.2),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.2))

    # Dados da GPU
    data_rect = mpatches.FancyBboxPatch((x, y - 1.5), 2.2, 1.1,
                                          boxstyle='round,pad=0.05', facecolor='#edc948',
                                          edgecolor='black', linewidth=1, alpha=0.7)
    ax.add_patch(data_rect)
    ax.text(x + 1.1, y - 0.95, f'Shard {i}\n(lote/4)', ha='center', va='center', fontsize=8)
    ax.annotate('', xy=(x + 1.1, y), xytext=(x + 1.1, y - 1.5 + 1.1),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.0))

# AllReduce
ax.annotate('', xy=(10.5, 5.75), xytext=(10.0, 5.75),
            arrowprops=dict(arrowstyle='->', color='navy', lw=1.5))
ax.text(6, 3.8, 'AllReduce: gradientes são agregados e\nsincroniados entre todas as GPUs',
         ha='center', fontsize=9, color='navy', style='italic',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='navy', alpha=0.7))

# Throughput
ax2 = axes[1]
n_gpus_list = [1, 2, 4, 8]
throughput = [1.0, 1.85, 3.4, 6.2]  # speedup realista (não linear)
ideal      = [1.0, 2.0,  4.0, 8.0]

ax2.plot(n_gpus_list, throughput, 'o-', color='#e15759', linewidth=2.5, markersize=8, label='Speedup real')
ax2.plot(n_gpus_list, ideal,      's--', color='#4e79a7', linewidth=1.5, markersize=6, label='Speedup ideal')
ax2.fill_between(n_gpus_list, throughput, ideal, alpha=0.1, color='red', label='Overhead (AllReduce)')
ax2.set_xlabel('Número de GPUs', fontsize=12)
ax2.set_ylabel('Speedup (relativo a 1 GPU)', fontsize=12)
ax2.set_title('MirroredStrategy: Speedup Real vs Ideal\n(overhead do AllReduce)', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(n_gpus_list)

for n, s in zip(n_gpus_list, throughput):
    ax2.text(n, s + 0.1, f'{s:.1f}×', ha='center', fontsize=10, fontweight='bold', color='#e15759')

plt.suptitle('tf.distribute.MirroredStrategy — Treinamento Distribuído Multi-GPU',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
