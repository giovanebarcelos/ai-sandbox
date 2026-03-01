# GO0633-Problema8LassoZerandoTodosOs
from sklearn.linear_model import LassoCV

# ❌ Alpha fixo muito alto:


if __name__ == "__main__":
    model = Lasso(alpha=1000.0)  # Muito alto!
    model.fit(X_train, y_train)
    print(f"Coefs não-zero: {sum(abs(model.coef_) > 1e-5)}")

    # ✅ Usar LassoCV para encontrar alpha ideal:
    model = LassoCV(alphas=np.logspace(-4, 1, 100), cv=5, random_state=42)
    model.fit(X_train, y_train)
    print(f"Melhor alpha: {model.alpha_:.4f}")
    print(f"Coefs não-zero: {sum(abs(model.coef_) > 1e-5)}")

    # Visualizar caminho de regularização:
    from sklearn.linear_model import lasso_path
    alphas, coefs, _ = lasso_path(X_train, y_train)

    import matplotlib.pyplot as plt
    plt.plot(alphas, coefs.T)
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Coefficients')
    plt.title('Lasso Path')
    plt.show()
