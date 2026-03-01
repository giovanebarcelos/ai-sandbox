# GO0623-Problema1RegressãoLinearComR²
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Verificar se há problemas:


if __name__ == "__main__":
    print(f"R²: {r2_score(y_test, y_pred)}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
    print(f"Média do target: {y_test.mean()}")
    print(f"Std do target: {y_test.std()}")

    # Comparar com baseline (média):
    baseline_pred = np.full(len(y_test), y_test.mean())
    baseline_r2 = r2_score(y_test, baseline_pred)
    print(f"R² do baseline: {baseline_r2}")  # Sempre 0.0
