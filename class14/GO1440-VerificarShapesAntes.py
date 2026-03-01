# GO1440-VerificarShapesAntes
# Verificar shapes antes de treinar
print(f"X_train shape: {X_train.shape}")  # Deve ser (N, 24, 1) e não (N, 24)

# Corrigir se necessário
if len(X_train.shape) == 2:
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
