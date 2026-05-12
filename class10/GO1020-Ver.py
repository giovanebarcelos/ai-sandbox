# GO1020-Ver
# Dicas de depuração: inspecionar labels, verificar o range dos dados e confirmar
# a função de perda correta para evitar erros silenciosos de configuração.
print(y_train[:10])                    # Ver se labels fazem sentido
print(X_train.min(), X_train.max())    # Range dos dados
model.compile(loss='sparse_categorical_crossentropy', ...)  # Loss correta
