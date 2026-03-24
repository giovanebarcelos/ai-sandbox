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

    # ═══════════════════════════════════════════════════════════════════
    # EXECUÇÃO COMPLETA: Treino, Validação, Teste e Uso Real
    # ═══════════════════════════════════════════════════════════════════
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report

    print("\n" + "═" * 60)
    print("EXECUÇÃO COMPLETA DO MODELO")
    print("═" * 60)

    # 1. Pré-processamento (normalização)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Aprende estatísticas do treino
    X_val_scaled = scaler.transform(X_val)          # Aplica no validation
    X_test_scaled = scaler.transform(X_test)        # Aplica no teste

    # 2. Treinar modelo
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train_scaled, y_train)
    print("\n✅ Modelo treinado com sucesso!")

    # 3. Avaliar no conjunto de validação (ajustar hiperparâmetros)
    y_val_pred = modelo.predict(X_val_scaled)
    acc_val = accuracy_score(y_val, y_val_pred)
    print(f"\n📊 Acurácia na Validação: {acc_val:.2%}")

    # 4. Avaliar no conjunto de teste (resultado final)
    y_test_pred = modelo.predict(X_test_scaled)
    acc_test = accuracy_score(y_test, y_test_pred)
    print(f"📊 Acurácia no Teste: {acc_test:.2%}")

    print("\n📋 Relatório de Classificação (Teste):")
    print(classification_report(y_test, y_test_pred, target_names=wine.target_names))

    # ═══════════════════════════════════════════════════════════════════
    # USO REAL: Classificar novos vinhos nunca vistos
    # ═══════════════════════════════════════════════════════════════════
    print("═" * 60)
    print("🍷 USO REAL: Classificando novos vinhos")
    print("═" * 60)

    # Simular 3 novos vinhos (dados fictícios baseados nas faixas do dataset)
    novos_vinhos = np.array([
        [13.2, 2.5, 2.4, 20, 100, 2.8, 3.0, 0.28, 2.0, 5.0, 1.0, 3.2, 1000],  # Vinho A
        [12.0, 3.5, 2.2, 18,  85, 1.5, 1.2, 0.45, 1.2, 4.0, 0.8, 2.0,  500],  # Vinho B
        [13.8, 1.8, 2.5, 17, 120, 2.5, 2.8, 0.22, 1.8, 6.5, 1.1, 3.5, 1200],  # Vinho C
    ])

    # Pré-processar com o mesmo scaler (IMPORTANTE!)
    novos_vinhos_scaled = scaler.transform(novos_vinhos)

    # Fazer predições
    predicoes = modelo.predict(novos_vinhos_scaled)
    probabilidades = modelo.predict_proba(novos_vinhos_scaled)

    print("\nResultados:")
    for i, (pred, probs) in enumerate(zip(predicoes, probabilidades)):
        classe = wine.target_names[pred]
        confianca = probs[pred] * 100
        print(f"  Vinho {chr(65+i)}: {classe} (confiança: {confianca:.1f}%)")
        print(f"           Probabilidades: {dict(zip(wine.target_names, [f'{p:.1%}' for p in probs]))}")

    print("\n✅ Pipeline completo executado com sucesso!")
