# GO0631-Problema7MeanSquaredErrorVs
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    # Gerar dados de exemplo
    X, y = make_regression(n_samples=200, n_features=5, noise=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Treinar modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("="*60)
    print("INTERPRETAÇÃO DE MÉTRICAS DE ERRO")
    print("="*60)

    # Calcular todas as métricas:
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    # RMSE está na mesma escala do target:
    print(f"\n📊 MÉTRICAS:")
    print(f"   RMSE: {rmse:.2f} (mesma unidade do target)")
    print(f"   MAE: {mae:.2f} (erro médio absoluto)")
    print(f"   Média do target: {y_test.mean():.2f}")

    # Regra de ouro:
    # RMSE < 10% da std(y) → modelo BOM
    # RMSE entre 10-20% → modelo OK
    # RMSE > 20% → modelo RUIM

    std_y = y_test.std()
    ratio = (rmse / std_y) * 100
    print(f"\n📏 COMPARAÇÃO:")
    print(f"   RMSE = {ratio:.1f}% do desvio padrão")

    if ratio < 10:
        print("   ✅ Modelo EXCELENTE")
    elif ratio < 20:
        print("   ⚠️ Modelo OK")
    else:
        print("   ❌ Modelo precisa melhorar")

    print("\n💡 REGRA DE OURO:")
    print("   • RMSE < 10% std(y) → modelo BOM")
    print("   • RMSE 10-20% std(y) → modelo OK")
    print("   • RMSE > 20% std(y) → modelo RUIM")

if __name__ == "__main__":
    main()
