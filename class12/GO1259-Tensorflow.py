# GO1259-Tensorflow
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    print("GPUs disponíveis:", gpus)
    print("TF versão:", tf.__version__)

    # ─── VISUALIZAÇÃO: DISPOSITIVOS DISPONÍVEIS ───
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Dispositivos
    all_devices = [
        ('CPU:0',   'CPU',    '#4e79a7'),
    ] + [(f'GPU:{i}', 'GPU', '#e15759') for i in range(len(gpus))]

    if not gpus:
        all_devices.append(('GPU', 'N/A', '#aaaaaa'))

    device_names = [d[0] for d in all_devices]
    device_colors = [d[2] for d in all_devices]
    device_vals = [1] * len(all_devices)

    bars = axes[0].bar(device_names, device_vals, color=device_colors, edgecolor='black', alpha=0.8)
    axes[0].set_title(f'Dispositivos Disponíveis\nTensorFlow {tf.__version__}', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Disponível')
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(['Não', 'Sim'])
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar, (name, dtype, _) in zip(bars, all_devices):
        status = 'Disponível' if dtype != 'N/A' else 'Não encontrada'
        axes[0].text(bar.get_x() + bar.get_width()/2, 0.5, status,
                     ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    # Comparação de velocidade de treinamento (simulado)
    modelos = ['ResNet50', 'VGG16', 'BERT', 'GPT-2']
    cpu_times  = [120, 180, 600, 3600]
    gpu_times  = [12,  18,  40,  180] if gpus else [None] * 4

    x = np.arange(len(modelos))
    width = 0.35
    axes[1].bar(x - width/2, cpu_times, width, label='CPU (min)',
                color='#4e79a7', edgecolor='black', alpha=0.8)
    if gpus:
        axes[1].bar(x + width/2, gpu_times, width, label='GPU (min)',
                    color='#e15759', edgecolor='black', alpha=0.8)
        axes[1].set_title('Tempo de Treinamento\nCPU vs GPU (estimativa por epoch)', fontsize=12)
    else:
        axes[1].set_title('Tempo de Treinamento na CPU\n(GPU não disponível)', fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(modelos, fontsize=10)
    axes[1].set_ylabel('Tempo por Época (seg)')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('TensorFlow — Informações de Hardware e Velocidade',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
