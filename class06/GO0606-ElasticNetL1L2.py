# GO0606-ElasticNetL1L2
# ═══════════════════════════════════════════════════════════════════
# ELASTIC NET (L1 + L2)
# ═══════════════════════════════════════════════════════════════════

from sklearn.linear_model import ElasticNet, ElasticNetCV

# ───────────────────────────────────────────────────────────────────
# ELASTIC NET CV
# ───────────────────────────────────────────────────────────────────

# l1_ratio controla mix entre L1 e L2:
#   l1_ratio=0 → Ridge puro (L2)
#   l1_ratio=1 → Lasso puro (L1)
#   l1_ratio=0.5 → 50% L1, 50% L2

l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]

elastic_cv = ElasticNetCV(
    l1_ratio=l1_ratios,
    alphas=alphas_lasso,
    cv=5,
    max_iter=10000
)

elastic_cv.fit(X_train_scaled, y_train)

print("="*60)
print("ELASTIC NET")
print("="*60)
print(f"Melhor alpha: {elastic_cv.alpha_:.3f}")
print(f"Melhor l1_ratio: {elastic_cv.l1_ratio_:.3f}")
print(f"  (0=Ridge, 1=Lasso, intermediário=mix)")
print(f"\nR² (treino): {elastic_cv.score(X_train_scaled, y_train):.3f}")
print(f"R² (teste):  {elastic_cv.score(X_test_scaled, y_test):.3f}")

# ───────────────────────────────────────────────────────────────────
# COMPARAR OS 3 MÉTODOS
# ───────────────────────────────────────────────────────────────────

from sklearn.linear_model import LinearRegression

models = {
    'Linear Regression': LinearRegression(),
    'Ridge': ridge_cv,
    'Lasso': lasso_cv,
    'Elastic Net': elastic_cv
}

print("\n" + "="*60)
print("COMPARAÇÃO FINAL")
print("="*60)
print(f"{'Modelo':<20} {'R² Treino':<12} {'R² Teste':<12} {'# Features'}")
print("-" * 60)

for name, model in models.items():
    if name == 'Linear Regression':
        model.fit(X_train_scaled, y_train)
        n_feat = X.shape[1]
    else:
        n_feat = np.sum(np.abs(model.coef_) > 1e-5)

    train_r2 = model.score(X_train_scaled, y_train)
    test_r2 = model.score(X_test_scaled, y_test)

    print(f"{name:<20} {train_r2:<12.3f} {test_r2:<12.3f} {n_feat}")

print("\n💡 Elastic Net combina vantagens de Ridge e Lasso:")
print("   - Regularização L2 lida bem com features correlacionadas")
print("   - Regularização L1 faz feature selection")
