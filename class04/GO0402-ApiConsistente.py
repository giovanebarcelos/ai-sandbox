# GO0402-ApiConsistente
# Demonstra a API CONSISTENTE do Scikit-learn:
# Todos os modelos seguem o mesmo padrão: fit(), predict(), score()

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Importar 4 modelos diferentes para demonstrar a consistência
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


if __name__ == "__main__":
    # ═══════════════════════════════════════════════════════════════════
    # CARREGAR E PREPARAR DADOS
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("API CONSISTENTE DO SCIKIT-LEARN")
    print("Todos os modelos usam: fit() → predict() → score()")
    print("=" * 60)

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )

    print(f"\n📊 Dataset: Iris")
    print(f"   Treino: {len(X_train)} amostras")
    print(f"   Teste:  {len(X_test)} amostras")
    print(f"   Classes: {list(iris.target_names)}")

    # ═══════════════════════════════════════════════════════════════════
    # DEFINIR MODELOS - Todos seguem a MESMA API!
    # ═══════════════════════════════════════════════════════════════════
    modelos = {
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
        "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
        "Support Vector Machine": SVC(kernel='rbf', random_state=42),
        "Naive Bayes": GaussianNB(),
    }

    print("\n" + "=" * 60)
    print("TREINANDO E AVALIANDO MODELOS")
    print("=" * 60)

    resultados = {}

    for nome, modelo in modelos.items():
        print(f"\n🔹 {nome}")

        # 1. TREINAR - Mesmo método para TODOS os modelos!
        modelo.fit(X_train, y_train)
        print("   ✅ modelo.fit(X_train, y_train)")

        # 2. PREVER - Mesmo método para TODOS os modelos!
        y_pred = modelo.predict(X_test)
        print("   ✅ modelo.predict(X_test)")

        # 3. AVALIAR - Mesmo método para TODOS os modelos!
        acuracia = modelo.score(X_test, y_test)
        print(f"   ✅ modelo.score(X_test, y_test) = {acuracia:.2%}")

        resultados[nome] = acuracia

    # ═══════════════════════════════════════════════════════════════════
    # RESUMO COMPARATIVO
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("📊 RESUMO - COMPARAÇÃO DOS MODELOS")
    print("=" * 60)

    # Ordenar por acurácia
    for nome, acc in sorted(resultados.items(), key=lambda x: x[1], reverse=True):
        barra = "█" * int(acc * 30)
        print(f"{nome:25} | {barra} {acc:.2%}")

    # Melhor modelo
    melhor = max(resultados, key=resultados.get)
    print(f"\n🏆 Melhor modelo: {melhor} ({resultados[melhor]:.2%})")

    # ═══════════════════════════════════════════════════════════════════
    # DEMONSTRAÇÃO: Predição com novo dado
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("🔮 PREDIÇÃO COM NOVO DADO")
    print("=" * 60)

    # Uma nova flor para classificar
    nova_flor = [[5.1, 3.5, 1.4, 0.2]]  # Medidas típicas de Iris-setosa
    print(f"Nova flor: sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2")

    print("\nPredições de cada modelo:")
    for nome, modelo in modelos.items():
        pred = modelo.predict(nova_flor)[0]
        classe = iris.target_names[pred]
        print(f"   {nome:25} → {classe}")

    print("\n✅ Demonstração concluída!")
