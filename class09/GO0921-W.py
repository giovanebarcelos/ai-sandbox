# GO0921-W
# ═══════════════════════════════════════════════════════════════════
# INICIALIZAÇÃO HE NORMAL — IMPLEMENTAÇÃO E COMPARAÇÃO
# Slide 26: Inicialização de Pesos
# ═══════════════════════════════════════════════════════════════════
"""
He (Kaiming) initialization — recomendado para camadas com ReLU:
  W = np.random.randn(n_in, n_out) * sqrt(2/n_in)

Garante que a variância dos gradientes se mantém estável durante
backpropagation em redes profundas com ReLU.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def relu(z):
    return np.maximum(0, z)


def simular_forward(W_init_fn, n_layers=10, n_neurons=256, n_samples=100):
    """
    Simula forward pass em rede profunda para verificar se os
    gradientes explodem ou somem conforme a inicialização.
    """
    x = np.random.randn(n_samples, n_neurons)
    stds = [x.std()]
    for _ in range(n_layers):
        W = W_init_fn(n_neurons, n_neurons)
        x = relu(x @ W)
        stds.append(x.std())
    return stds


if __name__ == "__main__":
    np.random.seed(42)
    n_in, n_out = 784, 256

    # He initialization (fórmula principal deste arquivo)
    W = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
    print("=" * 55)
    print("HE INITIALIZATION")
    print("=" * 55)
    print(f"  n_in={n_in}, n_out={n_out}")
    print(f"  std esperado: {np.sqrt(2/n_in):.4f}")
    print(f"  std real:     {W.std():.4f}")

    # Comparar propagação de sinal em rede profunda
    print("\nSimulando forward pass em rede 10 camadas...")
    stds_random = simular_forward(lambda ni,no: np.random.randn(ni,no))
    stds_he     = simular_forward(lambda ni,no: np.random.randn(ni,no)*np.sqrt(2/ni))
    stds_xavier = simular_forward(lambda ni,no: np.random.randn(ni,no)*np.sqrt(2/(ni+no)))

    print("\nStd das ativações por camada:")
    print(f"  {'Camada':>6}  {'Random':>10}  {'He':>10}  {'Xavier':>10}")
    for i in range(min(6, len(stds_he))):
        print(f"  {i:6d}  {stds_random[i]:10.4f}  "
              f"{stds_he[i]:10.4f}  {stds_xavier[i]:10.4f}")

    plt.figure(figsize=(8, 4))
    plt.plot(stds_random, "o-", label="Random N(0,1)", color="red")
    plt.plot(stds_he,     "s-", label="He (sqrt(2/n))", color="green")
    plt.plot(stds_xavier, "^-", label="Xavier", color="blue")
    plt.xlabel("Camada")
    plt.ylabel("Std das ativações (ReLU)")
    plt.title("Estabilidade do sinal: He vs Xavier vs Random")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("GO0921_he_initialization.png", dpi=100, bbox_inches="tight")
    plt.show()
    print("\nSalvo: GO0921_he_initialization.png")
    print("\n✅ He initialization mantém std estável em redes profundas com ReLU")
