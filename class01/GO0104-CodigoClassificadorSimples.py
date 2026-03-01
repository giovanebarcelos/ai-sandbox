# GO0104-CodigoClassificadorSimples
# Importar bibliotecas
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Carregar dados (150 amostras, 3 espécies)


if __name__ == "__main__":
    iris = load_iris()
    X = iris.data  # 4 features: comprimento/largura sépala e pétala
    y = iris.target  # 0=setosa, 1=versicolor, 2=virginica

    # 2. Dividir: 70% treino, 30% teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 3. Criar e treinar modelo
    modelo = DecisionTreeClassifier(max_depth=3)
    modelo.fit(X_train, y_train)

    # 4. Fazer predições
    y_pred = modelo.predict(X_test)

    # 5. Avaliar
    acuracia = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acuracia*100:.2f}%")  # ~95%
