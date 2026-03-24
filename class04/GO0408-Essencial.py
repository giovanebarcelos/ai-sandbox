# GO0408-Essencial
# Pipeline COMPLETO de Machine Learning - Código Essencial da Aula 04
# Demonstra todos os conceitos-chave: divisão, normalização, treino, avaliação

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np


if __name__ == "__main__":
    print("=" * 65)
    print("🎓 AULA 04 - CÓDIGO ESSENCIAL: Pipeline Completo de ML")
    print("=" * 65)

    # ═══════════════════════════════════════════════════════════════════
    # 1. CARREGAR DADOS
    # ═══════════════════════════════════════════════════════════════════
    print("\n📊 1. CARREGANDO DADOS")
    print("-" * 40)

    iris = load_iris()
    X = iris.data
    y = iris.target

    print(f"Dataset: Iris")
    print(f"Total de amostras: {len(X)}")
    print(f"Número de features: {X.shape[1]}")
    print(f"Features: {iris.feature_names}")
    print(f"Classes: {list(iris.target_names)}")
    print(f"Distribuição: {np.bincount(y)}")

    # ═══════════════════════════════════════════════════════════════════
    # 2. DIVIDIR DADOS (Treino/Teste - 80/20)
    # ═══════════════════════════════════════════════════════════════════
    print("\n✂️ 2. DIVISÃO DOS DADOS")
    print("-" * 40)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Treino: {len(X_train)} amostras ({len(X_train)/len(X)*100:.0f}%)")
    print(f"Teste:  {len(X_test)} amostras ({len(X_test)/len(X)*100:.0f}%)")
    print(f"Distribuição treino: {np.bincount(y_train)}")
    print(f"Distribuição teste:  {np.bincount(y_test)}")

    # ═══════════════════════════════════════════════════════════════════
    # 3. NORMALIZAR (StandardScaler)
    # ═══════════════════════════════════════════════════════════════════
    print("\n📏 3. NORMALIZAÇÃO (StandardScaler)")
    print("-" * 40)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Aprende e aplica no treino
    X_test_scaled = scaler.transform(X_test)         # Só aplica no teste

    print("Antes da normalização:")
    print(f"  X_train - média: {X_train.mean(axis=0).round(2)}")
    print(f"  X_train - std:   {X_train.std(axis=0).round(2)}")
    print("\nDepois da normalização:")
    print(f"  X_train - média: {X_train_scaled.mean(axis=0).round(2)}")
    print(f"  X_train - std:   {X_train_scaled.std(axis=0).round(2)}")

    # ═══════════════════════════════════════════════════════════════════
    # 4. TREINAR MODELO
    # ═══════════════════════════════════════════════════════════════════
    print("\n🧠 4. TREINAMENTO DO MODELO")
    print("-" * 40)

    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train_scaled, y_train)

    print(f"Modelo: DecisionTreeClassifier")
    print(f"Hiperparâmetros: max_depth=3")
    print("✅ Modelo treinado com sucesso!")

    # ═══════════════════════════════════════════════════════════════════
    # 5. AVALIAR MODELO
    # ═══════════════════════════════════════════════════════════════════
    print("\n📈 5. AVALIAÇÃO DO MODELO")
    print("-" * 40)

    # Predições
    y_pred = model.predict(X_test_scaled)

    # Acurácia
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia no teste: {accuracy:.2%}")

    # Validação Cruzada (K-Fold)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"\nValidação Cruzada (5-Fold):")
    print(f"  Scores: {cv_scores.round(3)}")
    print(f"  Média:  {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

    # Relatório de Classificação
    print(f"\n📋 Relatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # Matriz de Confusão
    print("🔢 Matriz de Confusão:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                 Predito")
    print(f"               setosa versicolor virginica")
    for i, classe in enumerate(iris.target_names):
        print(f"  Real {classe:10} {cm[i]}")

    # ═══════════════════════════════════════════════════════════════════
    # 6. USAR MODELO (Predição com novos dados)
    # ═══════════════════════════════════════════════════════════════════
    print("\n🔮 6. USO DO MODELO (Predição)")
    print("-" * 40)

    # Simular novas flores para classificar
    novas_flores = np.array([
        [5.1, 3.5, 1.4, 0.2],  # Típica setosa
        [6.0, 2.7, 4.5, 1.5],  # Típica versicolor
        [6.5, 3.0, 5.5, 2.0],  # Típica virginica
    ])

    # Normalizar com o mesmo scaler!
    novas_flores_scaled = scaler.transform(novas_flores)

    # Predizer
    predicoes = model.predict(novas_flores_scaled)

    print("Novas flores para classificar:")
    for i, (flor, pred) in enumerate(zip(novas_flores, predicoes)):
        classe = iris.target_names[pred]
        print(f"  Flor {i+1}: {flor} → {classe}")

    # ═══════════════════════════════════════════════════════════════════
    # RESUMO FINAL
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("✅ PIPELINE COMPLETO EXECUTADO COM SUCESSO!")
    print("=" * 65)
    print("""
📋 Conceitos demonstrados:
   ✓ Carregamento de dados (load_iris)
   ✓ Divisão treino/teste (train_test_split com stratify)
   ✓ Normalização (StandardScaler - fit no treino, transform no teste)
   ✓ Treinamento (model.fit)
   ✓ Avaliação (accuracy, cross_val_score, classification_report)
   ✓ Predição (model.predict com novos dados normalizados)
""")
