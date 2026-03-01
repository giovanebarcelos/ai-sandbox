# GO0616-ParaAnálise
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carregar e preparar dados
housing = fetch_california_housing()
X, y = housing.data, housing.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Normalizar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinar modelo
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# Gerar predições e resíduos
y_pred = model.predict(X_test_scaled)
residuos = y_test - y_pred

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Resíduos vs Predições
axes[0,0].scatter(y_pred, residuos, alpha=0.6)
axes[0,0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0,0].set_xlabel('Valores Preditos')
axes[0,0].set_ylabel('Resíduos')
axes[0,0].set_title('Resíduos vs Predições')
axes[0,0].grid(True, alpha=0.3)

# 2. Q-Q Plot
stats.probplot(residuos, dist="norm", plot=axes[0,1])
axes[0,1].set_title('Q-Q Plot')
axes[0,1].grid(True, alpha=0.3)

# 3. Histograma
axes[1,0].hist(residuos, bins=30, edgecolor='black', alpha=0.7)
axes[1,0].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1,0].set_xlabel('Resíduos')
axes[1,0].set_ylabel('Frequência')
axes[1,0].set_title('Distribuição dos Resíduos')
axes[1,0].grid(True, alpha=0.3)

# 4. Scale-Location
import numpy as np
std_residuos = np.sqrt(np.abs((residuos - residuos.mean()) / residuos.std()))
axes[1,1].scatter(y_pred, std_residuos, alpha=0.6)
axes[1,1].set_xlabel('Valores Preditos')
axes[1,1].set_ylabel('√|Resíduos Padronizados|')
axes[1,1].set_title('Scale-Location')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Testes estatísticos
from scipy.stats import shapiro, jarque_bera
import warnings

print("="*60)
print("TESTES DE NORMALIDADE DOS RESÍDUOS")
print("="*60)

# Shapiro-Wilk (usar subset se dataset for grande)
# ⚠️ Shapiro-Wilk não é confiável para N > 5000
if len(residuos) > 5000:
    print(f"\n⚠️  Dataset grande (N={len(residuos)}). Usando subset de 5000 amostras para Shapiro-Wilk.")
    residuos_sample = np.random.choice(residuos, size=5000, replace=False)
else:
    residuos_sample = residuos

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=UserWarning)
    shapiro_stat, shapiro_p = shapiro(residuos_sample)

print(f"\nShapiro-Wilk Test (N={len(residuos_sample)}):")
print(f"  Estatística: {shapiro_stat:.4f}")
print(f"  P-valor: {shapiro_p:.4f}")
print(f"  {'✅ Normal (p > 0.05)' if shapiro_p > 0.05 else '❌ Não-normal (p ≤ 0.05)'}")

# Jarque-Bera (funciona bem para qualquer tamanho)
jb_stat, jb_p = jarque_bera(residuos)
print(f"\nJarque-Bera Test (N={len(residuos)}):")
print(f"  Estatística: {jb_stat:.4f}")
print(f"  P-valor: {jb_p:.4f}")
print(f"  {'✅ Normal (p > 0.05)' if jb_p > 0.05 else '❌ Não-normal (p ≤ 0.05)'}")
