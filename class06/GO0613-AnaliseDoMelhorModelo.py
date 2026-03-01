# GO0613-AnáliseDoMelhorModelo
# ═══════════════════════════════════════════════════════════════════
# ANÁLISE DO MELHOR MODELO
# ═══════════════════════════════════════════════════════════════════

# Treinar melhor modelo (baseado em results)
best_model = Ridge(alpha=1.0)
best_model.fit(X_train_scaled, y_train)

y_pred = best_model.predict(X_test_scaled)
residuals = y_test - y_pred

# ───────────────────────────────────────────────────────────────────
# IMPORTÂNCIA DAS FEATURES
# ───────────────────────────────────────────────────────────────────

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': np.abs(best_model.coef_)
}).sort_values('Coefficient', ascending=False)

print("="*60)
print("TOP 10 FEATURES MAIS IMPORTANTES")
print("="*60)
print(feature_importance.head(10).to_string(index=False))

# Plotar
plt.figure(figsize=(10, 8))
top_n = 15
plt.barh(range(top_n), feature_importance['Coefficient'].values[:top_n])
plt.yticks(range(top_n), feature_importance['Feature'].values[:top_n])
plt.xlabel('|Coeficiente|')
plt.title(f'Top {top_n} Features Mais Importantes')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# ───────────────────────────────────────────────────────────────────
# ANÁLISE DE RESÍDUOS
# ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Resíduos vs Predições
axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('Predições')
axes[0, 0].set_ylabel('Resíduos')
axes[0, 0].set_title('Resíduos vs Predições')
axes[0, 0].grid(True, alpha=0.3)

# Distribuição dos resíduos
axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Resíduos')
axes[0, 1].set_ylabel('Frequência')
axes[0, 1].set_title('Distribuição dos Resíduos')
axes[0, 1].grid(True, alpha=0.3)

# Real vs Predito
axes[1, 0].scatter(y_test, y_pred, alpha=0.5)
lims = [y_test.min(), y_test.max()]
axes[1, 0].plot(lims, lims, 'r--', linewidth=2)
axes[1, 0].set_xlabel('Valores Reais')
axes[1, 0].set_ylabel('Valores Preditos')
axes[1, 0].set_title('Real vs Predito')
axes[1, 0].grid(True, alpha=0.3)

# Q-Q Plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ───────────────────────────────────────────────────────────────────
# ESTATÍSTICAS FINAIS
# ───────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("ESTATÍSTICAS FINAIS DO MODELO")
print("="*60)
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"\nMédia dos resíduos: {residuals.mean():.6f}")
print(f"Std dos resíduos: {residuals.std():.4f}")

# Teste de normalidade
stat, p = stats.shapiro(residuals[:5000])
print(f"\nTeste de Normalidade (Shapiro-Wilk):")
print(f"  P-valor: {p:.4f}")
if p > 0.05:
    print("  ✅ Resíduos normalmente distribuídos")
else:
    print("  ⚠️  Resíduos podem não ser normais")

print("\n✅ Análise completa!")
