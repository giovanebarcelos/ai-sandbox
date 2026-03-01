# GO0620-TabelaComparativa
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
import numpy as np

# Calcular todas as métricas
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

print("="*60)
print("MÉTRICAS DE AVALIAÇÃO")
print("="*60)
print(f"MAE:   {mae:>10.2f}  (unidade: mesma de y)")
print(f"MSE:   {mse:>10.2f}  (unidade: y²)")
print(f"RMSE:  {rmse:>10.2f}  (unidade: mesma de y)")
print(f"R²:    {r2:>10.4f}  (0-1, quanto maior melhor)")
print(f"MAPE:  {mape:>10.2f}% (erro percentual)")

# Visualização
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Real vs Predito
axes[0,0].scatter(y_test, y_pred, alpha=0.6)
axes[0,0].plot([y_test.min(), y_test.max()], 
               [y_test.min(), y_test.max()], 
               'r--', lw=2, label='Ideal')
axes[0,0].set_xlabel('Valor Real')
axes[0,0].set_ylabel('Valor Predito')
axes[0,0].set_title(f'Real vs Predito (R²={r2:.3f})')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 2. Distribuição dos Erros Absolutos
erros_abs = np.abs(y_test - y_pred)
axes[0,1].hist(erros_abs, bins=30, edgecolor='black', alpha=0.7)
axes[0,1].axvline(mae, color='r', linestyle='--', linewidth=2, 
                  label=f'MAE={mae:.2f}')
axes[0,1].set_xlabel('Erro Absoluto')
axes[0,1].set_ylabel('Frequência')
axes[0,1].set_title('Distribuição dos Erros')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# 3. Resíduos
residuos = y_test - y_pred
axes[1,0].scatter(y_pred, residuos, alpha=0.6)
axes[1,0].axhline(0, color='r', linestyle='--', linewidth=2)
axes[1,0].set_xlabel('Valor Predito')
axes[1,0].set_ylabel('Resíduo')
axes[1,0].set_title(f'Resíduos (RMSE={rmse:.2f})')
axes[1,0].grid(True, alpha=0.3)

# 4. Erro Percentual
erro_pct = 100 * np.abs((y_test - y_pred) / y_test)
axes[1,1].hist(erro_pct, bins=30, edgecolor='black', alpha=0.7)
axes[1,1].axvline(mape, color='r', linestyle='--', linewidth=2,
                  label=f'MAPE={mape:.2f}%')
axes[1,1].set_xlabel('Erro Percentual (%)')
axes[1,1].set_ylabel('Frequência')
axes[1,1].set_title('Distribuição do Erro Percentual')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
