# GO0810-TreinarSomNoDatasetDeVinhos
# ═══════════════════════════════════════════════════════════════════
# TREINAR SOM NO DATASET DE VINHOS
# ═══════════════════════════════════════════════════════════════════

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "minisom", "-q"])

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from minisom import MiniSom

# ───────────────────────────────────────────────────────────────────
# PREPARAR DADOS
# ───────────────────────────────────────────────────────────────────

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['class'] = wine.target


if __name__ == "__main__":
    X = df.drop('class', axis=1).values
    y = df['class'].values

    # Normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("="*60)
    print("TREINAMENTO DO SOM")
    print("="*60)

    # ───────────────────────────────────────────────────────────────────
    # CONFIGURAR E TREINAR SOM
    # ───────────────────────────────────────────────────────────────────

    # Grid 10x10 baseado em 5√n = 5√178 ≈ 67 → 8x8 ou 10x10
    som_x, som_y = 10, 10
    n_features = X_scaled.shape[1]

    print(f"\nGrid: {som_x}×{som_y} = {som_x*som_y} neurônios")
    print(f"Features: {n_features}")

    som = MiniSom(som_x, som_y, n_features,
                  sigma=5.0,  # max(10,10)/2 = 5
                  learning_rate=0.5,
                  neighborhood_function='gaussian',
                  random_seed=42)

    # Inicializar pesos
    som.random_weights_init(X_scaled)

    # Treinar
    n_iterations = 5000
    print(f"\nTreinando por {n_iterations} iterações...")
    som.train_random(X_scaled, n_iterations, verbose=True)

    # ───────────────────────────────────────────────────────────────────
    # AVALIAR CONVERGÊNCIA
    # ───────────────────────────────────────────────────────────────────

    qe = som.quantization_error(X_scaled)
    te = som.topographic_error(X_scaled)

    print("\n" + "="*60)
    print("MÉTRICAS DE QUALIDADE")
    print("="*60)
    print(f"Quantization Error: {qe:.4f}")
    print(f"  (menor = melhor, mede distância aos BMUs)")
    print(f"Topographic Error: {te:.4f}")
    print(f"  (menor = melhor, mede preservação de topologia)")

    # ───────────────────────────────────────────────────────────────────
    # MAPEAR VINHOS NO SOM
    # ───────────────────────────────────────────────────────────────────

    # Para cada vinho, encontrar seu BMU
    winners = np.array([som.winner(x) for x in X_scaled])

    # Adicionar coordenadas SOM ao dataframe
    df['som_x'] = winners[:, 0]
    df['som_y'] = winners[:, 1]

    print("\n✅ SOM treinado!")

    # ───────────────────────────────────────────────────────────────────
    # VISUALIZAÇÃO: U-MATRIX + MAPA DE CLASSES
    # ───────────────────────────────────────────────────────────────────

    class_names  = wine.target_names
    class_colors = ['#e74c3c', '#2ecc71', '#3498db']   # vermelho, verde, azul

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Gráfico 1: U-Matrix (fronteiras entre clusters) ─────────────
    ax1 = axes[0]
    umatrix = som.distance_map()          # valores [0, 1] normalizado
    im = ax1.imshow(umatrix.T, cmap='bone_r', origin='lower', aspect='auto')
    plt.colorbar(im, ax=ax1, label='Distância média aos vizinhos')

    # Sobreposição: cada vinho como ponto colorido pela classe
    for idx, (x_coord, y_coord) in enumerate(winners):
        cls = int(y[idx])
        ax1.plot(x_coord, y_coord, 'o',
                 color=class_colors[cls],
                 markersize=7, alpha=0.7,
                 markeredgecolor='white', markeredgewidth=0.5)

    ax1.set_title('U-Matrix — Fronteiras entre Clusters\n(claro = cluster, escuro = fronteira)', fontsize=12)
    ax1.set_xlabel('Neurônio x')
    ax1.set_ylabel('Neurônio y')

    legend_handles = [mpatches.Patch(color=class_colors[i], label=class_names[i])
                      for i in range(len(class_names))]
    ax1.legend(handles=legend_handles, loc='upper right', fontsize=10)

    # ── Gráfico 2: Mapa de calor de frequência por classe ───────────
    ax2 = axes[1]
    freq_maps = np.zeros((len(class_names), som_x, som_y))

    for idx, (x_coord, y_coord) in enumerate(winners):
        cls = int(y[idx])
        freq_maps[cls, x_coord, y_coord] += 1

    # Normalizar por célula
    total = freq_maps.sum(axis=0, keepdims=True)
    total[total == 0] = 1
    pct_maps = freq_maps / total   # proporção de cada classe por célula

    # Criar imagem RGB: mistura proporcional das cores
    rgb_colors = np.array([[0.91, 0.30, 0.24],   # vermelho  (class_0)
                            [0.18, 0.80, 0.44],   # verde     (class_1)
                            [0.20, 0.60, 0.86]])  # azul      (class_2)

    img = np.zeros((som_x, som_y, 3))
    for c in range(len(class_names)):
        img += pct_maps[c, :, :, np.newaxis] * rgb_colors[c]

    # Células sem nenhum vinho → cinza claro
    mask_empty = (total[0] == 0)
    img[mask_empty] = [0.9, 0.9, 0.9]

    ax2.imshow(np.clip(img, 0, 1).transpose(1, 0, 2), origin='lower', aspect='auto')
    ax2.set_title('Mapa de Classes — Dominância por Neurônio\n(cor = classe majoritária, mistura = sobreposição)', fontsize=12)
    ax2.set_xlabel('Neurônio x')
    ax2.set_ylabel('Neurônio y')
    ax2.legend(handles=legend_handles, loc='upper right', fontsize=10)

    # Contagem em cada célula
    for i in range(som_x):
        for j in range(som_y):
            n = int(total[0, i, j])
            if n > 0:
                ax2.text(i, j, str(n), ha='center', va='center',
                         fontsize=7, color='white',
                         fontweight='bold')

    plt.suptitle(f'SOM {som_x}×{som_y} — Dataset de Vinhos\n'
                 f'QE={qe:.4f}  TE={te:.4f}', fontsize=14)
    plt.tight_layout()
    plt.show()
