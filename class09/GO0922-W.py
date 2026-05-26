# GO0922-W
# ═══════════════════════════════════════════════════════════════════
# INICIALIZAÇÃO LeCun — PARA SIGMOID/SELU
# Slide 26: Inicialização de Pesos
# ═══════════════════════════════════════════════════════════════════
"""
LeCun initialization:
  W ~ N(0, sqrt(1/n_in))

Recomendado para: sigmoid em redes rasas, SELU.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def lecun_normal(n_in, n_out):
    """LeCun initialization (normal)."""
    return np.random.randn(n_in, n_out) * np.sqrt(1.0 / n_in)


def xavier_normal(n_in, n_out):
    return np.random.randn(n_in, n_out) * np.sqrt(2.0 / (n_in + n_out))


def he_normal(n_in, n_out):
    return np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)


if __name__ == "__main__":
    np.random.seed(42)
    n_in, n_out = 256, 128
    print("=" * 55)
    print("LeCun vs Xavier vs He — INICIALIZAÇÃO DE PESOS")
    print("=" * 55)
    print(f"n_in={n_in}, n_out={n_out}")
    print(f"  LeCun:   std = sqrt(1/n_in) = {np.sqrt(1/n_in):.4f}")
    print(f"  Xavier:  std = sqrt(2/(n_in+n_out)) = {np.sqrt(2/(n_in+n_out)):.4f}")
    print(f"  He:      std = sqrt(2/n_in) = {np.sqrt(2/n_in):.4f}")

    W_lecun  = lecun_normal(n_in, n_out)
    W_xavier = xavier_normal(n_in, n_out)
    W_he     = he_normal(n_in, n_out)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, (nome, W) in zip(axes, [("LeCun N(0,√(1/n))", W_lecun),
                                     ("Xavier N(0,√(2/(n+m)))", W_xavier),
                                     ("He N(0,√(2/n))", W_he)]):
        ax.hist(W.ravel(), bins=40, color="steelblue", edgecolor="white")
        ax.set_title(nome)
        ax.set_xlabel("Valor")
    plt.tight_layout()
    plt.savefig("GO0922_lecun_init.png", dpi=100, bbox_inches="tight")
    plt.show()
    print("\nSalvo: GO0922_lecun_init.png")
    print("\n📌 Quando usar cada uma:")
    print("  LeCun:  sigmoid / SELU")
    print("  Xavier: tanh / sigmoid (redes rasas)")
    print("  He:     ReLU / Leaky ReLU ← mais comum atualmente")
