# GO0629-Go0628problema5learningcurvesnãoconvergem
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, learning_curve
import matplotlib.pyplot as plt

def main():
    # Gerar dados de exemplo
    X, y = make_regression(n_samples=300, n_features=5, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print("="*60)
    print("DIAGNÓSTICO: LEARNING CURVES NÃO CONVERGEM")
    print("="*60)

    # Problema: Underfitting - modelo muito simples
    print("\n❌ PROBLEMA: Modelo Linear Simples (Underfitting)")
    model_simple = LinearRegression()
    train_sizes, train_scores, val_scores = learning_curve(
        model_simple, X_train, y_train, cv=5, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error'
    )
    print(f"   MSE Treino final: {-train_scores.mean(axis=1)[-1]:.2f}")
    print(f"   MSE Validação final: {-val_scores.mean(axis=1)[-1]:.2f}")
    print(f"   Gap grande = Underfitting!")

    # Solução 1: Adicionar complexidade com features polinomiais
    print("\n✅ SOLUÇÃO 1: Features Polinomiais (grau 2)")
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)

    model_poly = LinearRegression()
    train_sizes, train_scores, val_scores = learning_curve(
        model_poly, X_train_poly, y_train, cv=5, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error'
    )
    print(f"   MSE Treino final: {-train_scores.mean(axis=1)[-1]:.2f}")
    print(f"   MSE Validação final: {-val_scores.mean(axis=1)[-1]:.2f}")
    print(f"   Melhor convergência!")

    # Solução 2: Ridge com regularização (se houver overfitting)
    print("\n✅ SOLUÇÃO 2: Ridge com Regularização")
    model_ridge = Ridge(alpha=10.0)
    train_sizes, train_scores, val_scores = learning_curve(
        model_ridge, X_train_poly, y_train, cv=5, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error'
    )
    print(f"   MSE Treino final: {-train_scores.mean(axis=1)[-1]:.2f}")
    print(f"   MSE Validação final: {-val_scores.mean(axis=1)[-1]:.2f}")
    print(f"   Curvas convergindo = Bom modelo!")

    print("\n💡 DICA:")
    print("   • Gap grande + ambos erros altos → UNDERFITTING → adicionar features")
    print("   • Gap grande + erro treino baixo → OVERFITTING → adicionar regularização")
    print("   • Curvas convergindo → Modelo balanceado! ✅")

if __name__ == "__main__":
    main()
