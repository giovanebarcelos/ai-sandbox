# GO0810-TreinarSomNoDatasetDeVinhos
# ═══════════════════════════════════════════════════════════════════
# TREINAR SOM NO DATASET DE VINHOS
# ═══════════════════════════════════════════════════════════════════

from minisom import MiniSom

# ───────────────────────────────────────────────────────────────────
# PREPARAR DADOS
# ───────────────────────────────────────────────────────────────────


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
