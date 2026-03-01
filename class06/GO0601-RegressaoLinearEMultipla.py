# GO0601-RegressãoLinearEMúltipla
# ═══════════════════════════════════════════════════════════════════
# REGRESSÃO LINEAR SIMPLES E MÚLTIPLA
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ───────────────────────────────────────────────────────────────────
# EXEMPLO 1: Regressão Linear Simples (1 feature)
# ───────────────────────────────────────────────────────────────────

# Dados sintéticos: preço vs área
np.random.seed(42)
area = np.random.uniform(50, 300, 500)  # Mais pontos para dar estabilidade ao ajuste
preco = 100 + 2 * area + np.random.normal(0, 30, 500)  # Preço em mil reais

# Reshape para sklearn
X = area.reshape(-1, 1)
y = preco

# Treinar modelo
model = LinearRegression()
model.fit(X, y)

# Coeficientes
print("="*60)
print("REGRESSÃO LINEAR SIMPLES")
print("="*60)
print(f"Intercepto (β₀): R$ {model.intercept_:.2f}k")
print(f"Coeficiente (β₁): R$ {model.coef_[0]:.2f}k por m²")
print(f"Fórmula: preço = {model.intercept_:.2f} + ({model.coef_[0]:.4f} * área)")

# Predições
y_pred = model.predict(X)

# Métricas
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"\nMSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f} (erro médio em mil reais)")
print(f"MAE:  {mae:.2f}")
print(f"R²:   {r2:.4f} ({r2*100:.2f}% da variância explicada)")

# Visualização
plt.figure(figsize=(10, 6))
plt.scatter(area, preco, alpha=0.5, label='Dados reais')
plt.plot(area, y_pred, color='red', linewidth=2, label='Modelo')
plt.xlabel('Área (m²)', fontsize=12)
plt.ylabel('Preço (mil reais)', fontsize=12)
plt.title('Regressão Linear: Preço vs Área', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ───────────────────────────────────────────────────────────────────
# EXEMPLO 2: Regressão Linear Múltipla (múltiplas features)
# ───────────────────────────────────────────────────────────────────

# Dataset sintético mais complexo
n_samples = 1000  # Mais observações para regressão múltipla mais estável
area = np.random.uniform(50, 300, n_samples)
quartos = np.random.randint(1, 6, n_samples)
idade = np.random.uniform(0, 50, n_samples)
distancia = np.random.uniform(0, 20, n_samples)

# Preço com relação mais complexa
preco = (100 + 
         2 * area + 
         15 * quartos - 
         3 * idade - 
         5 * distancia + 
         np.random.normal(0, 40, n_samples))

# Criar DataFrame
df = pd.DataFrame({
    'area': area,
    'quartos': quartos,
    'idade': idade,
    'distancia': distancia,
    'preco': preco
})

print("\n" + "="*60)
print("REGRESSÃO LINEAR MÚLTIPLA")
print("="*60)
print(f"\nDados: {df.shape[0]} casas, {df.shape[1]-1} features")
print(f"\nPrimeiras linhas:")
print(df.head())

# Dividir dados
X = df[['area', 'quartos', 'idade', 'distancia']]
y = df['preco']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Treinar
model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

# Coeficientes
print("\n" + "="*60)
print("COEFICIENTES DO MODELO")
print("="*60)
print(f"Intercepto: R$ {model_multi.intercept_:.2f}k")
formula_terms = [f"{coef:.4f}*{feature}" for feature, coef in zip(X.columns, model_multi.coef_)]
print("Fórmula: preço = " + f"{model_multi.intercept_:.2f} + " + " + ".join(formula_terms))
for feature, coef in zip(X.columns, model_multi.coef_):
    sinal = "+" if coef >= 0 else ""
    print(f"{feature:12s}: {sinal}{coef:7.2f}k")

# Avaliar
y_train_pred = model_multi.predict(X_train)
y_test_pred = model_multi.predict(X_test)

print("\n" + "="*60)
print("MÉTRICAS")
print("="*60)
print(f"{'Métrica':<10} {'Treino':>10} {'Teste':>10}")
print("-" * 32)
print(f"{'RMSE':<10} {np.sqrt(mean_squared_error(y_train, y_train_pred)):>10.2f} "
      f"{np.sqrt(mean_squared_error(y_test, y_test_pred)):>10.2f}")
print(f"{'MAE':<10} {mean_absolute_error(y_train, y_train_pred):>10.2f} "
      f"{mean_absolute_error(y_test, y_test_pred):>10.2f}")
print(f"{'R²':<10} {r2_score(y_train, y_train_pred):>10.4f} "
      f"{r2_score(y_test, y_test_pred):>10.4f}")

# Predição de exemplo
exemplo = pd.DataFrame({
    'area': [150],
    'quartos': [3],
    'idade': [10],
    'distancia': [5]
})
pred_exemplo = model_multi.predict(exemplo)[0]
print(f"\n📍 Predição exemplo:")
print(f"   Casa: 150m², 3 quartos, 10 anos, 5km do centro")
print(f"   Preço estimado: R$ {pred_exemplo:.2f}k")
