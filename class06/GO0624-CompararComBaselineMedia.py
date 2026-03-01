# GO0624-CompararComBaselineMédia
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

def main():
    # Gerar dados de exemplo
    X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Normalizar dados (importante para evitar matriz mal condicionada)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Criar features polinomiais
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    # Treinar modelo com alpha maior para melhor condicionamento
    model = Ridge(alpha=10.0)
    model.fit(X_train_poly, y_train)

    # Avaliar
    train_score = model.score(X_train_poly, y_train)
    test_score = model.score(X_test_poly, y_test)

    print("="*60)
    print("MODELO RIDGE COM FEATURES POLINOMIAIS")
    print("="*60)
    print(f"R² Treino: {train_score:.4f}")
    print(f"R² Teste:  {test_score:.4f}")
    print(f"Alpha: {model.alpha}")
    print(f"Features originais: {X_train.shape[1]}")
    print(f"Features após polinomial: {X_train_poly.shape[1]}")

if __name__ == "__main__":
    main()
