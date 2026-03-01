# GO0619-GráficoDeImportância
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

# Carregar dados
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

# Treinar com dados normalizados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

# Criar DataFrame com coeficientes
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coef_Padronizado': model.coef_
})

# Ordenar por magnitude absoluta
coef_df['Abs_Coef'] = coef_df['Coef_Padronizado'].abs()
coef_df = coef_df.sort_values('Abs_Coef', ascending=True)

# Plotar
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['red' if x < 0 else 'green' for x in coef_df['Coef_Padronizado']]
ax.barh(coef_df['Feature'], coef_df['Coef_Padronizado'], color=colors, alpha=0.7)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('Coeficiente Padronizado', fontsize=12)
ax.set_title('Importância das Features (Coeficientes Padronizados)', fontsize=14)
ax.grid(axis='x', alpha=0.3)

# Adicionar valores
for i, (idx, row) in enumerate(coef_df.iterrows()):
    ax.text(row['Coef_Padronizado'], i, f" {row['Coef_Padronizado']:.3f}",
            va='center', fontsize=10)

plt.tight_layout()
plt.show()

print("="*60)
print("IMPORTÂNCIA DAS FEATURES")
print("="*60)
print(coef_df[['Feature', 'Coef_Padronizado']].to_string(index=False))
