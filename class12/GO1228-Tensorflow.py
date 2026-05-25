# GO1228-Tensorflow
from tensorflow.keras.layers import Conv2D
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass


if __name__ == "__main__":
    Conv2D(
        filters=32,           # Número de filtros (feature maps)
        kernel_size=(3, 3),   # Tamanho do kernel
        strides=(1, 1),       # Passo de deslocamento
        padding='valid',      # 'valid' ou 'same'
        activation='relu',    # Função de ativação
        input_shape=(28,28,1) # Apenas na 1ª camada
    )

    # ─── VISUALIZAÇÃO 1: OPERAÇÃO DE CONVOLUÇÃO ───
    # Simular um passo da convolução: janela 3×3 sobre input 5×5
    np.random.seed(0)
    input_map = np.array([
        [1, 0, 2, 1, 0],
        [0, 3, 1, 0, 2],
        [2, 1, 0, 3, 1],
        [1, 0, 2, 1, 0],
        [0, 1, 1, 2, 3],
    ], dtype=float)

    kernel = np.array([
        [ 1,  0, -1],
        [ 1,  0, -1],
        [ 1,  0, -1],
    ], dtype=float)

    # Convolução 'valid': sem padding → output menor que input
    # Fórmula: out_size = (in_size - kernel_size) / stride + 1
    # Aqui: (5 - 3) / 1 + 1 = 3  →  saída 3×3
    h, w = input_map.shape
    kh, kw = kernel.shape
    out_h, out_w = h - kh + 1, w - kw + 1  # dimensões da saída 'valid'
    feature_map_valid = np.zeros((out_h, out_w))
    for i in range(out_h):          # desliza verticalmente
        for j in range(out_w):      # desliza horizontalmente
            # Produto elemento a elemento entre janela e kernel, depois soma tudo
            feature_map_valid[i, j] = np.sum(input_map[i:i+kh, j:j+kw] * kernel)

    # Convolução 'same': padding de zeros → output IGUAL ao input
    # np.pad(input, 1): adiciona 1 linha/coluna de zeros em cada borda
    # Permite que as bordas da imagem também sejam processadas
    padded = np.pad(input_map, 1)   # input 5×5 → padded 7×7
    feature_map_same = np.zeros((h, w))  # saída mantém 5×5
    for i in range(h):
        for j in range(w):
            feature_map_same[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Input
    im0 = axes[0].imshow(input_map, cmap='Blues', vmin=-1, vmax=3)
    axes[0].set_title('Input\n(5×5×1)', fontsize=11, fontweight='bold')
    for i in range(5):
        for j in range(5):
            axes[0].text(j, i, f'{input_map[i,j]:.0f}', ha='center', va='center', fontsize=11)
    # Destacar a janela 3×3 na posição [0,0]
    rect = mpatches.Rectangle((-0.5, -0.5), 3, 3, linewidth=2.5, edgecolor='red', facecolor='none')
    axes[0].add_patch(rect)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_xlabel('Janela 3×3 (vermelho) = 1 passo', fontsize=8)

    # Kernel
    im1 = axes[1].imshow(kernel, cmap='RdBu', vmin=-1, vmax=1)
    axes[1].set_title('Kernel\n(3×3) — 1 filtro', fontsize=11, fontweight='bold')
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f'{kernel[i,j]:.0f}', ha='center', va='center',
                         fontsize=13, fontweight='bold',
                         color='white' if abs(kernel[i,j]) > 0.5 else 'black')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_xlabel('Detector de bordas verticais', fontsize=8)

    # Feature map valid
    im2 = axes[2].imshow(feature_map_valid, cmap='RdBu')
    axes[2].set_title('Feature Map\npadding="valid" (3×3)', fontsize=11, fontweight='bold')
    for i in range(out_h):
        for j in range(out_w):
            axes[2].text(j, i, f'{feature_map_valid[i,j]:.0f}', ha='center', va='center', fontsize=11)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].set_xlabel(f'Output: {out_h}×{out_w} (sem padding)', fontsize=8)

    # Feature map same
    im3 = axes[3].imshow(feature_map_same, cmap='RdBu')
    axes[3].set_title('Feature Map\npadding="same" (5×5)', fontsize=11, fontweight='bold')
    for i in range(h):
        for j in range(w):
            axes[3].text(j, i, f'{feature_map_same[i,j]:.0f}', ha='center', va='center', fontsize=9)
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    axes[3].set_xlabel('Output: 5×5 (preserva dimensão)', fontsize=8)

    plt.suptitle('Conv2D — Operação de Convolução: Input ✕ Kernel = Feature Map', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # ─── VISUALIZAÇÃO 2: IMPACTO DOS PARÂMETROS NA DIMENSÃO DE SAÍDA ───
    # Fórmula: out = floor((in + 2*pad - kernel) / stride) + 1
    input_size = 28
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Efeito do kernel_size (stride=1, padding='valid')
    kernel_sizes = [1, 3, 5, 7, 9, 11]
    out_sizes_k = [input_size - k + 1 for k in kernel_sizes]
    axes[0].bar(kernel_sizes, out_sizes_k, color='#4e79a7', edgecolor='black', width=1.5)
    axes[0].axhline(input_size, color='red', linestyle='--', label=f'Input={input_size}')
    for k, o in zip(kernel_sizes, out_sizes_k):
        axes[0].text(k, o + 0.3, str(o), ha='center', va='bottom', fontsize=10)
    axes[0].set_xlabel('kernel_size')
    axes[0].set_ylabel('Output size')
    axes[0].set_title('Efeito do kernel_size\n(stride=1, padding="valid")', fontsize=11)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Efeito do stride (kernel=3, padding='valid')
    strides = [1, 2, 3, 4]
    out_sizes_s = [(input_size - 3) // s + 1 for s in strides]
    axes[1].bar(strides, out_sizes_s, color='#f28e2b', edgecolor='black')
    for s, o in zip(strides, out_sizes_s):
        axes[1].text(s, o + 0.3, str(o), ha='center', va='bottom', fontsize=10)
    axes[1].set_xlabel('stride')
    axes[1].set_ylabel('Output size')
    axes[1].set_title('Efeito do stride\n(kernel=3, padding="valid")', fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')

    # Efeito de filters: número de feature maps na saída
    filter_counts = [8, 16, 32, 64, 128, 256]
    volumes = [26 * 26 * f for f in filter_counts]  # kernel=3, valid, 28×28 input
    bars = axes[2].bar(range(len(filter_counts)), [v / 1e3 for v in volumes],
                        color='#e15759', edgecolor='black')
    axes[2].set_xticks(range(len(filter_counts)))
    axes[2].set_xticklabels([f'filters={f}' for f in filter_counts], rotation=30, ha='right', fontsize=9)
    axes[2].set_ylabel('Volume de saída (×1000)')
    axes[2].set_title('Efeito de filters\n(input=28×28, kernel=3, valid)\nOutput shape: 26×26×filters', fontsize=10)
    for bar, v in zip(bars, volumes):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{v//1000}K', ha='center', va='bottom', fontsize=9)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Conv2D — Impacto dos Parâmetros na Dimensão da Saída', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

