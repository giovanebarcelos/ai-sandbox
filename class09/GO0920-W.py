# GO0920-W
# ═══════════════════════════════════════════════════════════════════
# INICIALIZAÇÃO DE PESOS — FÓRMULA XAVIER/GLOROT
# Slide 26: Inicialização de Pesos em Redes Neurais
# ═══════════════════════════════════════════════════════════════════
"""
Xavier/Glorot Initialization:
  W ~ N(0, sqrt(2 / (n_in + n_out)))
  ou Uniforme: W ~ U(-sqrt(6/(n_in+n_out)), +sqrt(6/(n_in+n_out)))

Recomendado para: sigmoid, tanh
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def xavier_normal(n_in, n_out):
    """Xavier initialization (normal)."""
    std = np.sqrt(2.0 / (n_in + n_out))
    return np.random.randn(n_in, n_out) * std


def xavier_uniform(n_in, n_out):
    """Xavier initialization (uniforme)."""
    limit = np.sqrt(6.0 / (n_in + n_out))
    return np.random.uniform(-limit, limit, (n_in, n_out))


def he_normal(n_in, n_out):
    """He initialization (para ReLU)."""
    std = np.sqrt(2.0 / n_in)
    return np.random.randn(n_in, n_out) * std


if __name__ == "__main__":
    np.random.seed(42)
    n_in, n_out = 128, 64

    W_xavier_n = xavier_normal(n_in, n_out)
    W_xavier_u = xavier_uniform(n_in, n_out)
    W_he       = he_normal(n_in, n_out)
    W_random   = np.random.randn(n_in, n_out)  # sem inicialização inteligente

    print("=" * 55)
    print("COMPARAÇÃO DE INICIALIZAÇÕES DE PESOS")
    print("=" * 55)
    for nome, W in [("Random N(0,1)", W_random),
                    ("Xavier Normal", W_xavier_n),
                    ("Xavier Uniforme", W_xavier_u),
                    ("He Normal (ReLU)", W_he)]:
        print(f"  {nome:20s}  std={W.std():.4f}  "
              f"min={W.min():.4f}  max={W.max():.4f}")

    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    for ax, (nome, W) in zip(axes, [("Random N(0,1)", W_random),
                                     ("Xavier Normal", W_xavier_n),
                                     ("Xavier Uniforme", W_xavier_u),
                                     ("He Normal (ReLU)", W_he)]):
        ax.hist(W.ravel(), bins=40, color="steelblue", edgecolor="white")
        ax.set_title(nome, fontsize=9)
        ax.set_xlabel("Valor do peso")
        ax.set_ylabel("Frequência")

    plt.suptitle("Distribuições de Inicialização de Pesos", fontsize=12)
    plt.tight_layout()
    plt.savefig("GO0920_inicializacoes.png", dpi=100, bbox_inches="tight")
    plt.show()
    print("\nSalvo: GO0920_inicializacoes.png")
    print("\n📌 Resumo:")
    print("  Xavier Normal:   std = sqrt(2/(n_in+n_out))")
    print("  Xavier Uniforme: limite = sqrt(6/(n_in+n_out))")
    print("  He Normal:       std = sqrt(2/n_in)  ← para ReLU")
