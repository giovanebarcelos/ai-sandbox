# GO0924-W
# ═══════════════════════════════════════════════════════════════════
# INICIALIZAÇÃO ALEATÓRIA SIMPLES (RANDOM SMALL)
# Slide 26: Problema com inicialização sem cuidado
# ═══════════════════════════════════════════════════════════════════
"""
Inicialização simples com escala:
  W ~ N(0, sqrt(1/n_in))

Equivalente ao LeCun, mas mostrando o problema de escala.
Pesos muito grandes ou muito pequenos quebram o aprendizado.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def relu(z):
    return np.maximum(0, z)


def testar_escala(scale, n_layers=8, n=128, n_samples=64):
    """Simula forward pass e mede std das ativações."""
    x = np.random.randn(n_samples, n)
    stds = []
    for _ in range(n_layers):
        W = np.random.randn(n, n) * scale
        x = relu(x @ W)
        stds.append(x.std())
        if stds[-1] == 0 or stds[-1] > 1e10:
            break
    return stds


if __name__ == "__main__":
    np.random.seed(42)
    n_in = 256

    print("=" * 55)
    print("EFEITO DA ESCALA NA INICIALIZAÇÃO DE PESOS")
    print("=" * 55)

    escalas = {
        "Muito pequena (0.001)": 0.001,
        "Pequena (0.01)":        0.01,
        "Correta sqrt(1/n)":     np.sqrt(1.0 / n_in),
        "He sqrt(2/n)":          np.sqrt(2.0 / n_in),
        "Grande (1.0)":          1.0,
    }

    plt.figure(figsize=(10, 4))
    for nome, scale in escalas.items():
        stds = testar_escala(scale)
        plt.plot(stds, "o-", label=f"{nome} (scale={scale:.4f})", alpha=0.8)
        final = stds[-1] if stds else 0
        print(f"  {nome:30s}: std final = {final:.6f}")

    plt.xlabel("Camada")
    plt.ylabel("Std das ativações")
    plt.title("Efeito da Escala na Inicialização (ReLU, 8 camadas)")
    plt.legend(fontsize=8, loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("GO0924_escala_inicializacao.png", dpi=100, bbox_inches="tight")
    plt.show()
    print("\nSalvo: GO0924_escala_inicializacao.png")
    print("\n📌 Escalas muito pequenas → ativações somem (vanishing)")
    print("  Escalas muito grandes → ativações explodem (exploding)")
    print("  He/Xavier → mantém escala estável por toda a rede")
