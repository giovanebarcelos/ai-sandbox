# GO0402-ApiConsistente
from sklearn.XXX import ModeloYYY

# 1. Criar instância


if __name__ == "__main__":
    modelo = ModeloYYY(hiperparametro=valor)

    # 2. Treinar
    modelo.fit(X_train, y_train)

    # 3. Prever
    y_pred = modelo.predict(X_test)

    # 4. Avaliar
    from sklearn.metrics import accuracy_score
    acuracia = accuracy_score(y_test, y_pred)
