# GO0401-RegrasDeOuro
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Exemplo: Dataset de vinhos
# X = features (13 características químicas)
# y = target (qualidade: 0, 1, 2)

# Carregar dados
from sklearn.datasets import load_wine


if __name__ == "__main__":
    wine = load_wine()
    X = wine.data
    y = wine.target

    print(f"Total de amostras: {len(X)}")
    print(f"Features: {wine.feature_names}")
    print(f"Classes: {wine.target_names}")

    # ═══════════════════════════════════════════════════════════════════
    # MÉTODO 1: Train-Test Split (80-20)
    # ═══════════════════════════════════════════════════════════════════

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,        # 20% para teste
        random_state=42,      # Reproduzibilidade
        stratify=y            # Manter proporção de classes
    )

    print(f"\nTreino: {len(X_train)} amostras")
    print(f"Teste: {len(X_test)} amostras")

    # Verificar proporção de classes
    print("\nDistribuição original:", np.bincount(y) / len(y))
    print("Distribuição treino:", np.bincount(y_train) / len(y_train))
    print("Distribuição teste:", np.bincount(y_test) / len(y_test))

    # ═══════════════════════════════════════════════════════════════════
    # MÉTODO 2: Train-Validation-Test Split (60-20-20)
    # ═══════════════════════════════════════════════════════════════════

    # Primeiro: separa treino+val (80%) e teste (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Segundo: separa treino (75% de temp = 60% total) e val (25% de temp = 20% total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    print(f"\n3-Way Split:")
    print(f"Treino: {len(X_train)} ({len(X_train)/len(X)*100:.0f}%)")
    print(f"Validação: {len(X_val)} ({len(X_val)/len(X)*100:.0f}%)")
    print(f"Teste: {len(X_test)} ({len(X_test)/len(X)*100:.0f}%)")

    # ═══════════════════════════════════════════════════════════════════
    # IMPORTANTE: Processar DEPOIS de dividir!
    # ═══════════════════════════════════════════════════════════════════

    # ❌ ERRADO (data leakage):
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)  # Usa estatísticas de TODO dataset
    # X_train, X_test = train_test_split(X_scaled, ...)

    # ✅ CERTO:
    # X_train, X_test = train_test_split(X, ...)
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)  # Aprende só do treino
    # X_test_scaled = scaler.transform(X_test)  # Aplica transformação do treino
