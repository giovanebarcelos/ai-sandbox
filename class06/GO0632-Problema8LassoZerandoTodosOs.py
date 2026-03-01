# GO0632-Problema8LassoZerandoTodosOs
from sklearn.linear_model import Lasso
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    # Gerar dados de exemplo
    X, y = make_regression(n_samples=200, n_features=10, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print("="*60)
    print("DEMONSTRAÇÃO: LASSO ZERANDO COEFICIENTES")
    print("="*60)

    # Treinar modelo com alpha muito alto
    model = Lasso(alpha=1000.0)  # Alpha muito alto!
    model.fit(X_train, y_train)

    print(f"\n❌ PROBLEMA: Alpha muito alto (alpha=1000.0)")
    print(f"   Coeficientes: {model.coef_}")
    print(f"   Coefs não-zero: {sum(abs(model.coef_) > 1e-5)}")
    print(f"   Todos os coeficientes foram zerados!")

if __name__ == "__main__":
    main()
