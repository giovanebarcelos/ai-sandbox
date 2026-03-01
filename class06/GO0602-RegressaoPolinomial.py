# GO0602-RegressãoPolinomial
# ═══════════════════════════════════════════════════════════════════
# REGRESSÃO POLINOMIAL
# ═══════════════════════════════════════════════════════════════════

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt

# Gerar dados sintéticos não-lineares
np.random.seed(42)
X = np.sort(np.random.rand(50) * 10).reshape(-1, 1)
y = 2 + 0.5*X.ravel() + 3*np.sin(X.ravel()) + np.random.randn(50) * 0.5

# ───────────────────────────────────────────────────────────────────
# COMPARAR DIFERENTES GRAUS
# ───────────────────────────────────────────────────────────────────

degrees = [1, 3, 9, 15]
X_plot = np.linspace(0, 10, 300).reshape(-1, 1)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for i, degree in enumerate(degrees):
    # Criar pipeline: PolynomialFeatures + LinearRegression
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    y_plot = model.predict(X_plot)

    # Plotar
    axes[i].scatter(X, y, alpha=0.6, label='Dados', color='blue')
    axes[i].plot(X_plot, y_plot, 'r-', label=f'Grau {degree}', linewidth=2)
    axes[i].set_title(f'Polinômio Grau {degree}')
    axes[i].set_xlabel('X')
    axes[i].set_ylabel('y')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

    # Scores
    train_score = model.score(X, y)
    axes[i].text(0.05, 0.95, f'R² = {train_score:.3f}', 
                 transform=axes[i].transAxes, 
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

print("="*60)
print("ANÁLISE")
print("="*60)
print("Grau 1:  Underfitting (linha reta, não captura padrão)")
print("Grau 3:  Bom ajuste (captura padrão sem overfit)")
print("Grau 9:  Começando overfit (oscilações desnecessárias)")
print("Grau 15: Overfitting severo (memoriza ruído)")
