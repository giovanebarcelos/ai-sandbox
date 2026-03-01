# GO0509-MnistPreparaçãoDosDados
# ═══════════════════════════════════════════════════════════════════
# MNIST - PREPARAÇÃO DOS DADOS
# ═══════════════════════════════════════════════════════════════════

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    print("="*60)
    print("PREPARAÇÃO DOS DADOS")
    print("="*60)

    # ───────────────────────────────────────────────────────────────────
    # SUBSET PARA TREINO MAIS RÁPIDO (opcional)
    # ───────────────────────────────────────────────────────────────────

    # MNIST completo tem 70k imagens - pode demorar
    # Vamos usar subset para demonstração
    SUBSET_SIZE = 10000  # Use 70000 para dataset completo

    X_subset = X[:SUBSET_SIZE]
    y_subset = y[:SUBSET_SIZE]

    print(f"\nUsando {SUBSET_SIZE} exemplos para treino/teste")

    # ───────────────────────────────────────────────────────────────────
    # DIVISÃO TREINO/TESTE
    # ───────────────────────────────────────────────────────────────────

    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y_subset, 
        test_size=0.2, 
        random_state=42,
        stratify=y_subset  # Manter proporção de classes
    )

    print(f"\nDivisão:")
    print(f"  Treino: {X_train.shape[0]} exemplos")
    print(f"  Teste:  {X_test.shape[0]} exemplos")

    # ───────────────────────────────────────────────────────────────────
    # NORMALIZAÇÃO
    # ───────────────────────────────────────────────────────────────────

    # Pixels vão de 0-255, vamos normalizar para 0-1
    # Isso ajuda o KNN (distâncias) e convergência de algoritmos

    print("\nNormalizando pixels [0-255] → [0-1]...")

    X_train_normalized = X_train / 255.0
    X_test_normalized = X_test / 255.0

    # Alternativamente, StandardScaler (média 0, std 1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("  ✅ Normalização simples (/ 255)")
    print("  ✅ StandardScaler (μ=0, σ=1)")

    # ───────────────────────────────────────────────────────────────────
    # VERIFICAR NORMALIZAÇÃO
    # ───────────────────────────────────────────────────────────────────

    print("\nEstatísticas após normalização simples:")
    print(f"  Min: {X_train_normalized.min().min():.3f}")
    print(f"  Max: {X_train_normalized.max().max():.3f}")
    print(f"  Média: {X_train_normalized.mean().mean():.3f}")

    print("\nEstatísticas após StandardScaler:")
    print(f"  Média: {X_train_scaled.mean():.3f}")
    print(f"  Std: {X_train_scaled.std():.3f}")

    print("\n✅ Dados preparados para treinamento!")
