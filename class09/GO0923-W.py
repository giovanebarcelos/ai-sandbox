# GO0923-W
# ═══════════════════════════════════════════════════════════════════
# INICIALIZAÇÃO HE UNIFORME
# Slide 26: Inicialização de Pesos
# ═══════════════════════════════════════════════════════════════════
"""
He Uniform initialization:
  W = np.random.randn(n_in, n_out) * sqrt(2/n_in)

Versão uniforme:
  limite = sqrt(6/n_in)
  W ~ Uniform(-limite, +limite)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def he_normal(n_in, n_out):
    """He Normal: W ~ N(0, sqrt(2/n_in))"""
    return np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)


def he_uniform(n_in, n_out):
    """He Uniform: W ~ U(-sqrt(6/n_in), +sqrt(6/n_in))"""
    limit = np.sqrt(6.0 / n_in)
    return np.random.uniform(-limit, limit, (n_in, n_out))


if __name__ == "__main__":
    np.random.seed(42)
    n_in, n_out = 784, 256

    # He Normal (fórmula principal)
    W_normal  = he_normal(n_in, n_out)
    W_uniform = he_uniform(n_in, n_out)

    print("=" * 55)
    print("HE INITIALIZATION — Normal vs Uniforme")
    print("=" * 55)
    print(f"  He Normal:   std={W_normal.std():.4f}  "
          f"(esperado {np.sqrt(2/n_in):.4f})")
    print(f"  He Uniforme: std={W_uniform.std():.4f}  "
          f"(esperado {np.sqrt(6/n_in)/np.sqrt(3):.4f})")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].hist(W_normal.ravel(),  bins=50, color="steelblue", edgecolor="white")
    axes[0].set_title("He Normal: sqrt(2/n_in)")
    axes[1].hist(W_uniform.ravel(), bins=50, color="darkorange", edgecolor="white")
    axes[1].set_title("He Uniforme: sqrt(6/n_in)")
    for ax in axes:
        ax.set_xlabel("Valor do peso"); ax.set_ylabel("Frequência")
    plt.tight_layout()
    plt.savefig("GO0923_he_uniform.png", dpi=100, bbox_inches="tight")
    plt.show()
    print("\nSalvo: GO0923_he_uniform.png")
    print("\n✅ He Normal é mais comum; Uniforme é ligeiramente mais estável")
