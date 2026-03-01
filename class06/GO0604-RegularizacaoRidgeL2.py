# GO0604-RegularizaçãoRidgeL2
# ═══════════════════════════════════════════════════════════════════
# REGULARIZAÇÃO RIDGE (L2)
# ═══════════════════════════════════════════════════════════════════

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Gerar dados com multicolinearidade
np.random.seed(42)
n_samples = 100
X1 = np.random.randn(n_samples)
X2 = X1 + np.random.randn(n_samples) * 0.1  # Fortemente correlacionado com X1
X3 = np.random.randn(n_samples)
X = np.column_stack([X1, X2, X3])
y = 3*X1 - 2*X3 + np.random.randn(n_samples) * 0.5

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Normalizar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ───────────────────────────────────────────────────────────────────
# TESTAR DIFERENTES VALORES DE ALPHA
# ───────────────────────────────────────────────────────────────────

alphas = np.logspace(-3, 3, 50)
coefs = []
train_scores = []
test_scores = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    coefs.append(ridge.coef_)
    train_scores.append(ridge.score(X_train_scaled, y_train))
    test_scores.append(ridge.score(X_test_scaled, y_test))

coefs = np.array(coefs)

# ───────────────────────────────────────────────────────────────────
# PLOTAR COEFICIENTES vs ALPHA
# ───────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Coeficientes
for i in range(X.shape[1]):
    ax1.plot(alphas, coefs[:, i], label=f'Feature {i+1}')
ax1.set_xscale('log')
ax1.set_xlabel('Alpha (λ)')
ax1.set_ylabel('Coeficiente')
ax1.set_title('Coeficientes vs Regularização')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# R² scores
ax2.plot(alphas, train_scores, label='Treino', linewidth=2)
ax2.plot(alphas, test_scores, label='Teste', linewidth=2)
ax2.set_xscale('log')
ax2.set_xlabel('Alpha (λ)')
ax2.set_ylabel('R²')
ax2.set_title('Performance vs Regularização')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Marcar melhor alpha
best_idx = np.argmax(test_scores)
best_alpha = alphas[best_idx]
ax2.axvline(x=best_alpha, color='r', linestyle='--', 
            label=f'Melhor α={best_alpha:.3f}')
ax2.legend()

plt.tight_layout()
plt.show()

# ───────────────────────────────────────────────────────────────────
# RIDGE CV (encontra melhor alpha automaticamente)
# ───────────────────────────────────────────────────────────────────

ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train_scaled, y_train)

print("="*60)
print("RIDGE REGRESSION")
print("="*60)
print(f"Melhor alpha (CV): {ridge_cv.alpha_:.3f}")
print(f"R² (treino): {ridge_cv.score(X_train_scaled, y_train):.3f}")
print(f"R² (teste):  {ridge_cv.score(X_test_scaled, y_test):.3f}")
print(f"\nCoeficientes:")
for i, coef in enumerate(ridge_cv.coef_):
    print(f"  Feature {i+1}: {coef:.3f}")
