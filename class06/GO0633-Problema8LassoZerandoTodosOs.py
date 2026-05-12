# GO0633-Problema8LassoZerandoTodosOs
from sklearn.linear_model import LassoCV

# ❌ Alpha fixo muito alto:

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

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

    plt.plot(alphas, coefs.T)
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Coefficients')
    plt.title('Lasso Path')
    plt.show()
