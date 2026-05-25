# GO1260-Tensorflow
# Treina 2x mais rápido em GPUs modernas
from tensorflow.keras.mixed_precision import set_global_policy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass


if __name__ == '__main__':
    set_global_policy('mixed_float16')

    # ─── VISUALIZAÇÃO: PRECISÃO MISTA FP16 vs FP32 ───
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Memória por tipo
    tipos = ['FP32\n(float32)', 'FP16\n(float16)', 'BF16\n(bfloat16)']
    bits  = [32, 16, 16]
    expoente = [8, 5, 8]
    mantissa = [23, 10, 7]
    colors_base = ['#e15759', '#4e79a7', '#59a14f']

    bars = axes[0].bar(tipos, bits, color=colors_base, edgecolor='black', alpha=0.8)
    axes[0].set_ylabel('Bits por Número')
    axes[0].set_title('Memória: FP32 vs FP16 vs BF16\n(FP16/BF16 usam 50% menos)', fontsize=10)
    for bar, b in zip(bars, bits):
        axes[0].text(bar.get_x() + bar.get_width()/2, b + 0.5, f'{b}-bit',
                      ha='center', fontsize=11, fontweight='bold')
    axes[0].set_ylim(0, 40)
    axes[0].grid(True, alpha=0.3, axis='y')

    # 2. Throughput (simulado)
    hardware = ['V100', 'A100', 'RTX 3090']
    fp32_tflops = [14,  19.5, 35.6]
    fp16_tflops = [112, 312,  142]
    x = np.arange(len(hardware))
    w = 0.35
    axes[1].bar(x - w/2, fp32_tflops, w, label='FP32', color='#e15759', edgecolor='black', alpha=0.8)
    axes[1].bar(x + w/2, fp16_tflops, w, label='FP16', color='#4e79a7', edgecolor='black', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(hardware)
    axes[1].set_ylabel('Performance (TFLOPS)')
    axes[1].set_title('FP32 vs FP16 Performance\n(TFLOPS por GPU)', fontsize=10)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    # 3. Fluxo de dado na precisão mista
    axes[2].axis('off')
    fluxo = [
        ('Weights\n(FP32)', '#e15759'),
        ('Cast para FP16\n(forward pass)', '#f28e2b'),
        ('Forward\n(FP16)', '#4e79a7'),
        ('Gradients\n(FP16)', '#59a14f'),
        ('Loss Scaling\n(x1024)', '#edc948'),
        ('Update Weights\n(FP32)', '#e15759'),
    ]
    for i, (label, color) in enumerate(fluxo):
        y = 5.5 - i * 1.0
        import matplotlib.patches as mpatches
        rect = mpatches.FancyBboxPatch((1.5, y - 0.3), 4, 0.55,
                                        boxstyle='round,pad=0.05', facecolor=color,
                                        edgecolor='black', alpha=0.8)
        axes[2].add_patch(rect)
        axes[2].text(3.5, y, label, ha='center', va='center', fontsize=9, fontweight='bold',
                     color='white')
        if i < len(fluxo) - 1:
            axes[2].annotate('', xy=(3.5, y - 0.3),
                              xytext=(3.5, y - 0.62 if i == 0 else y - 0.62),
                              arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))
    axes[2].set_xlim(0, 7)
    axes[2].set_ylim(-0.2, 6)
    axes[2].set_title('Fluxo Mixed Precision\n(FP16 forward + FP32 weights)', fontsize=10, fontweight='bold')

    plt.suptitle('Mixed Precision Training (mixed_float16) — Conceito e Vantagens',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
