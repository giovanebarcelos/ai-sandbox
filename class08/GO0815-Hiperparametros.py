# GO0815-Hiperparâmetros
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "minisom", "-q"])

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom


if __name__ == "__main__":
    # Carregar e preparar dados
    wine = load_wine()
    scaler = StandardScaler()
    X = scaler.fit_transform(wine.data)
    n_features = X.shape[1]   # 13 features

    som = MiniSom(10, 10, n_features, sigma=5.0, learning_rate=0.5)
    som.random_weights_init(X)
    som.train_random(X, 5000)

    new_sample = scaler.transform(wine.data[[0]])   # exemplo: 1ª amostra
    winner = som.winner(new_sample[0])  # Mapear novo dado
    print(f"Novo dado mapeado para neurônio: {winner}")

    # ───────────────────────────────────────────────────────────────────
    # GRÁFICO: QE × Hiperparâmetros (sigma e learning_rate)
    # ───────────────────────────────────────────────────────────────────

    sigmas         = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0]
    learning_rates = [0.1, 0.3, 0.5, 0.7, 1.0]
    n_iter         = 2000

    qe_matrix = np.zeros((len(sigmas), len(learning_rates)))

    print("\nCalculando QE para cada combinação de hiperparâmetros...")
    for i, sigma in enumerate(sigmas):
        for j, lr in enumerate(learning_rates):
            s = MiniSom(10, 10, n_features,
                        sigma=sigma, learning_rate=lr,
                        random_seed=42)
            s.random_weights_init(X)
            s.train_random(X, n_iter, verbose=False)
            qe_matrix[i, j] = s.quantization_error(X)
        print(f"  sigma={sigma:.1f} ✓")

    # Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # ── Heatmap ──────────────────────────────────────────────────────
    ax1 = axes[0]
    im = ax1.imshow(qe_matrix, cmap='RdYlGn_r', aspect='auto')
    plt.colorbar(im, ax=ax1, label='Quantization Error')
    ax1.set_xticks(range(len(learning_rates)))
    ax1.set_xticklabels([str(lr) for lr in learning_rates])
    ax1.set_yticks(range(len(sigmas)))
    ax1.set_yticklabels([str(s) for s in sigmas])
    ax1.set_xlabel('Learning Rate')
    ax1.set_ylabel('Sigma')
    ax1.set_title('QE por Hiperparâmetros\n(verde = melhor, vermelho = pior)')

    for i in range(len(sigmas)):
        for j in range(len(learning_rates)):
            ax1.text(j, i, f'{qe_matrix[i, j]:.2f}',
                     ha='center', va='center', fontsize=8,
                     color='black')

    # ── Curvas de QE × sigma para cada learning_rate ─────────────────
    ax2 = axes[1]
    for j, lr in enumerate(learning_rates):
        ax2.plot(sigmas, qe_matrix[:, j], 'o-', label=f'lr={lr}', linewidth=1.8)

    ax2.set_xlabel('Sigma')
    ax2.set_ylabel('Quantization Error')
    ax2.set_title('QE × Sigma por Learning Rate')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('GO0815 — Impacto dos Hiperparâmetros no SOM (Wine Dataset)', fontsize=13)
    plt.tight_layout()
    plt.show()

