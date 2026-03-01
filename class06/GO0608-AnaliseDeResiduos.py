# GO0608-AnáliseDeResíduos
# ═══════════════════════════════════════════════════════════════════
# ANÁLISE DE RESÍDUOS
# ═══════════════════════════════════════════════════════════════════

from scipy import stats
import matplotlib.pyplot as plt

# Treinar modelo
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predições
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Resíduos
residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred

# ───────────────────────────────────────────────────────────────────
# PLOTS DE DIAGNÓSTICO
# ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Resíduos vs Valores Preditos
axes[0, 0].scatter(y_train_pred, residuals_train, alpha=0.6, label='Treino')
axes[0, 0].scatter(y_test_pred, residuals_test, alpha=0.6, label='Teste')
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('Valores Preditos')
axes[0, 0].set_ylabel('Resíduos')
axes[0, 0].set_title('Resíduos vs Predições')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Histograma dos Resíduos
axes[0, 1].hist(residuals_train, bins=30, alpha=0.7, label='Treino', density=True)
axes[0, 1].hist(residuals_test, bins=20, alpha=0.7, label='Teste', density=True)
# Sobrepor distribuição normal
mu, std = residuals_train.mean(), residuals_train.std()
x = np.linspace(residuals_train.min(), residuals_train.max(), 100)
axes[0, 1].plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2, label='Normal')
axes[0, 1].set_xlabel('Resíduos')
axes[0, 1].set_ylabel('Densidade')
axes[0, 1].set_title('Distribuição dos Resíduos')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Q-Q Plot (teste de normalidade)
stats.probplot(residuals_train, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot (Normalidade dos Resíduos)')
axes[1, 0].grid(True, alpha=0.3)

# 4. Valores Reais vs Preditos
axes[1, 1].scatter(y_train, y_train_pred, alpha=0.6, label='Treino')
axes[1, 1].scatter(y_test, y_test_pred, alpha=0.6, label='Teste')
# Linha perfeita (y=x)
lims = [min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())]
axes[1, 1].plot(lims, lims, 'r--', linewidth=2, label='Predição perfeita')
axes[1, 1].set_xlabel('Valores Reais')
axes[1, 1].set_ylabel('Valores Preditos')
axes[1, 1].set_title('Real vs Predito')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ───────────────────────────────────────────────────────────────────
# TESTES ESTATÍSTICOS
# ───────────────────────────────────────────────────────────────────

print("="*60)
print("ANÁLISE DE RESÍDUOS")
print("="*60)

print(f"\nEstatísticas dos Resíduos:")
print(f"  Média: {residuals_train.mean():.6f} (deve ser ~0)")
print(f"  Mediana: {np.median(residuals_train):.6f}")
print(f"  Std: {residuals_train.std():.3f}")

# Teste de normalidade (Shapiro-Wilk)
statistic, p_value = stats.shapiro(residuals_train[:5000])  # Máx 5000 amostras
print(f"\nTeste de Normalidade (Shapiro-Wilk):")
print(f"  Estatística: {statistic:.4f}")
print(f"  P-valor: {p_value:.4f}")
if p_value > 0.05:
    print("  ✅ Resíduos parecem normalmente distribuídos")
else:
    print("  ⚠️  Resíduos podem não ser normais")

# Teste de homocedasticidade (variância constante)
# Dividir em 2 grupos e comparar variâncias
mid = len(y_train_pred) // 2
idx_sorted = np.argsort(y_train_pred)
group1 = residuals_train[idx_sorted[:mid]]
group2 = residuals_train[idx_sorted[mid:]]

statistic, p_value = stats.levene(group1, group2)
print(f"\nTeste de Homocedasticidade (Levene):")
print(f"  Estatística: {statistic:.4f}")
print(f"  P-valor: {p_value:.4f}")
if p_value > 0.05:
    print("  ✅ Variância constante (homocedasticidade)")
else:
    print("  ⚠️  Heterocedasticidade detectada")
