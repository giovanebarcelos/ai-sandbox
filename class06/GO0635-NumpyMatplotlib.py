# GO0635-NumpyMatplotlib
# GO06EX2-RegularizacaoRidgeLasso
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

print("="*70)
print("EXERCÍCIO 2: REGULARIZAÇÃO - RIDGE VS LASSO")
print("="*70)

# ═══════════════════════════════════════════════════════════════════
# 1. GERAR DADOS COM MULTICOLINEARIDADE
# ═══════════════════════════════════════════════════════════════════

np.random.seed(42)
n_samples = 200
n_features = 10

# Gerar features correlacionadas
X = np.random.randn(n_samples, n_features)

# Criar multicolinearidade: algumas features são combinações de outras
X[:, 5] = X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1
X[:, 6] = X[:, 2] * 2 + np.random.randn(n_samples) * 0.1
X[:, 7] = X[:, 3] + X[:, 4] + np.random.randn(n_samples) * 0.1

# Target: apenas 4 features são realmente importantes
coef_verdadeiros = np.array([3.0, -2.0, 5.0, 1.5, 0, 0, 0, 0, 0, 0])
y = X @ coef_verdadeiros + np.random.randn(n_samples) * 2

print("\n📊 DADOS GERADOS:")
print(f"   • {n_samples} amostras, {n_features} features")
print(f"   • 4 features importantes, 6 irrelevantes")
print(f"   • Multicolinearidade presente")
print(f"   • Coeficientes verdadeiros: {coef_verdadeiros}")

# Split e normalização
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ═══════════════════════════════════════════════════════════════════
# 2. TREINAR MODELOS COM DIFERENTES ALPHAS
# ═══════════════════════════════════════════════════════════════════

alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
resultados_ridge = {}
resultados_lasso = {}

print("\n🔄 TREINANDO MODELOS...")

# Modelo base (sem regularização)
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
mse_lr = mean_squared_error(y_test, y_pred_lr)

print(f"\n📐 LINEAR REGRESSION (sem regularização):")
print(f"   • MSE Teste: {mse_lr:.4f}")
print(f"   • Coeficientes: {lr.coef_}")
print(f"   • Features não-zero: {np.sum(np.abs(lr.coef_) > 0.01)}")

# Ridge
print(f"\n🔵 RIDGE REGRESSION (L2):")
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    y_pred = ridge.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)

    resultados_ridge[alpha] = {
        'model': ridge,
        'coef': ridge.coef_,
        'mse': mse,
        'non_zero': np.sum(np.abs(ridge.coef_) > 0.01)
    }

    print(f"   Alpha={alpha:6.3f}: MSE={mse:.4f}, "
          f"Features não-zero={resultados_ridge[alpha]['non_zero']}")

# Lasso
print(f"\n🔴 LASSO REGRESSION (L1):")
for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    y_pred = lasso.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)

    resultados_lasso[alpha] = {
        'model': lasso,
        'coef': lasso.coef_,
        'mse': mse,
        'non_zero': np.sum(np.abs(lasso.coef_) > 0.01)
    }

    print(f"   Alpha={alpha:6.3f}: MSE={mse:.4f}, "
          f"Features não-zero={resultados_lasso[alpha]['non_zero']}")

# ═══════════════════════════════════════════════════════════════════
# 3. VISUALIZAÇÃO: COEFICIENTES VS ALPHA
# ═══════════════════════════════════════════════════════════════════

print("\n📊 GERANDO VISUALIZAÇÕES...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Ridge
coefs_ridge = np.array([resultados_ridge[a]['coef'] for a in alphas])
for i in range(n_features):
    ax1.plot(alphas, coefs_ridge[:, i], 'o-', label=f'Feature {i}', 
            linewidth=2, markersize=6)

ax1.set_xscale('log')
ax1.set_xlabel('Alpha (Regularization Strength)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Coefficient Value', fontsize=12, fontweight='bold')
ax1.set_title('Ridge (L2): Coeficientes vs Alpha', fontsize=14, fontweight='bold')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)

# Lasso
coefs_lasso = np.array([resultados_lasso[a]['coef'] for a in alphas])
for i in range(n_features):
    ax2.plot(alphas, coefs_lasso[:, i], 's-', label=f'Feature {i}', 
            linewidth=2, markersize=6)

ax2.set_xscale('log')
ax2.set_xlabel('Alpha (Regularization Strength)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Coefficient Value', fontsize=12, fontweight='bold')
ax2.set_title('Lasso (L1): Coeficientes vs Alpha', fontsize=14, fontweight='bold')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)

plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════
# 4. COMPARAÇÃO DE SELEÇÃO DE FEATURES
# ═══════════════════════════════════════════════════════════════════

print("\n📈 SELEÇÃO DE FEATURES...")

fig, ax = plt.subplots(figsize=(12, 6))

# Plot número de features não-zero vs alpha
non_zero_ridge = [resultados_ridge[a]['non_zero'] for a in alphas]
non_zero_lasso = [resultados_lasso[a]['non_zero'] for a in alphas]

ax.plot(alphas, non_zero_ridge, 'o-', linewidth=3, markersize=10, 
       label='Ridge (L2)', color='blue')
ax.plot(alphas, non_zero_lasso, 's-', linewidth=3, markersize=10, 
       label='Lasso (L1)', color='red')

ax.set_xscale('log')
ax.set_xlabel('Alpha (Regularization Strength)', fontsize=12, fontweight='bold')
ax.set_ylabel('Número de Features Não-Zero', fontsize=12, fontweight='bold')
ax.set_title('Seleção Automática de Features: Ridge vs Lasso', 
            fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_yticks(range(0, 11))

plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════
# 5. HEATMAP DE COEFICIENTES
# ═══════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Ridge heatmap
sns.heatmap(coefs_ridge.T, annot=True, fmt='.2f', cmap='RdBu_r', 
           center=0, ax=ax1, cbar_kws={'label': 'Coefficient Value'},
           xticklabels=[f'{a:.3f}' for a in alphas],
           yticklabels=[f'Feature {i}' for i in range(n_features)])
ax1.set_title('Ridge (L2): Heatmap de Coeficientes', fontsize=14, fontweight='bold')
ax1.set_xlabel('Alpha', fontsize=12)

# Lasso heatmap
sns.heatmap(coefs_lasso.T, annot=True, fmt='.2f', cmap='RdBu_r', 
           center=0, ax=ax2, cbar_kws={'label': 'Coefficient Value'},
           xticklabels=[f'{a:.3f}' for a in alphas],
           yticklabels=[f'Feature {i}' for i in range(n_features)])
ax2.set_title('Lasso (L1): Heatmap de Coeficientes', fontsize=14, fontweight='bold')
ax2.set_xlabel('Alpha', fontsize=12)

plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════
# 6. CONCLUSÕES E INTERPRETAÇÃO
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("🎓 CONCLUSÕES DO EXERCÍCIO:")
print("="*70)

print("""
📚 DIFERENÇAS PRINCIPAIS:

🔵 RIDGE (L2 Regularization):
   • NUNCA zera coeficientes completamente
   • Reduz magnitude de TODOS os coeficientes
   • Mantém todas as features, mas com pesos menores
   • Melhor quando TODAS as features são relevantes
   • Lida bem com multicolinearidade
   • Penalidade: ∑(βᵢ²)

🔴 LASSO (L1 Regularization):
   • ZERA coeficientes de features irrelevantes
   • Realiza SELEÇÃO AUTOMÁTICA de features
   • Produz modelos ESPARSOS (sparse)
   • Melhor quando muitas features são irrelevantes
   • Pode arbitrariamente escolher entre features correlacionadas
   • Penalidade: ∑|βᵢ|

⚖️ QUANDO USAR QUAL:

✅ USE RIDGE quando:
   • Todas as features são potencialmente úteis
   • Multicolinearidade é um problema
   • Interpretabilidade não é crítica
   • Quer suavizar todos os coeficientes

✅ USE LASSO quando:
   • Muitas features são irrelevantes
   • Quer seleção automática de features
   • Quer modelo interpretável e esparso
   • Precisa reduzir dimensionalidade

💡 DICA PROFISSIONAL:
   Use ELASTIC NET (combinação de L1 + L2) para ter o melhor dos dois!

   from sklearn.linear_model import ElasticNet
   model = ElasticNet(alpha=1.0, l1_ratio=0.5)  # 50% L1 + 50% L2
""")

print("="*70)
print("✅ EXERCÍCIO 2 COMPLETO!")
print("="*70)
