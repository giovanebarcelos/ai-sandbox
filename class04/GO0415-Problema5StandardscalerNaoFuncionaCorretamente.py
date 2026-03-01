# GO0415-Problema5StandardscalerNãoFuncionaCorretamente
from sklearn.preprocessing import StandardScaler

# ❌ ERRADO:


if __name__ == "__main__":
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)  # ❌ FIT NO TESTE!

    # ✅ CORRETO:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # FIT + TRANSFORM no treino
    X_test_scaled = scaler.transform(X_test)         # Só TRANSFORM no teste

    # Lembrar de salvar o scaler para produção:
    import joblib
    joblib.dump(scaler, 'scaler.pkl')
