# GO0605-RegularizaçãoLassoL1
# ═══════════════════════════════════════════════════════════════════
# REGULARIZAÇÃO LASSO (L1)
# ═══════════════════════════════════════════════════════════════════

from sklearn.linear_model import Lasso, LassoCV
import matplotlib.pyplot as plt

# Usar mesmos dados do slide anterior

# ───────────────────────────────────────────────────────────────────
# TESTAR DIFERENTES VALORES DE ALPHA
# ───────────────────────────────────────────────────────────────────

alphas_lasso = np.logspace(-3, 1, 50)
coefs_lasso = []
train_scores_lasso = []
test_scores_lasso = []
n_features_used = []

for alpha in alphas_lasso:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    coefs_lasso.append(lasso.coef_)
    train_scores_lasso.append(lasso.score(X_train_scaled, y_train))
    test_scores_lasso.append(lasso.score(X_test_scaled, y_test))
    # Contar features com coef != 0
    n_features_used.append(np.sum(np.abs(lasso.coef_) > 1e-5))

coefs_lasso = np.array(coefs_lasso)

# ───────────────────────────────────────────────────────────────────
# PLOTAR
# ───────────────────────────────────────────────────────────────────

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Coeficientes
for i in range(X.shape[1]):
    ax1.plot(alphas_lasso, coefs_lasso[:, i], label=f'Feature {i+1}')
ax1.set_xscale('log')
ax1.set_xlabel('Alpha (λ)')
ax1.set_ylabel('Coeficiente')
ax1.set_title('LASSO: Coeficientes vs Regularização')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# R² scores
ax2.plot(alphas_lasso, train_scores_lasso, label='Treino', linewidth=2)
ax2.plot(alphas_lasso, test_scores_lasso, label='Teste', linewidth=2)
ax2.set_xscale('log')
ax2.set_xlabel('Alpha (λ)')
ax2.set_ylabel('R²')
ax2.set_title('Performance vs Regularização')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Número de features usadas
ax3.plot(alphas_lasso, n_features_used, 'o-', color='green', linewidth=2)
ax3.set_xscale('log')
ax3.set_xlabel('Alpha (λ)')
ax3.set_ylabel('# Features (coef ≠ 0)')
ax3.set_title('Feature Selection por LASSO')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ───────────────────────────────────────────────────────────────────
# LASSO CV
# ───────────────────────────────────────────────────────────────────

lasso_cv = LassoCV(alphas=alphas_lasso, cv=5, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)

print("="*60)
print("LASSO REGRESSION")
print("="*60)
print(f"Melhor alpha (CV): {lasso_cv.alpha_:.3f}")
print(f"R² (treino): {lasso_cv.score(X_train_scaled, y_train):.3f}")
print(f"R² (teste):  {lasso_cv.score(X_test_scaled, y_test):.3f}")
print(f"\nCoeficientes:")
for i, coef in enumerate(lasso_cv.coef_):
    if np.abs(coef) > 1e-5:
        print(f"  Feature {i+1}: {coef:.3f}")
    else:
        print(f"  Feature {i+1}: 0 (eliminada)")

print(f"\nFeatures selecionadas: {np.sum(np.abs(lasso_cv.coef_) > 1e-5)}/{len(lasso_cv.coef_)}")
