# GO0413 - Problema 4: DataConversionWarning -- y deve ser 1D
# AVISO: sklearn espera y como array 1D, nao coluna (n, 1)
# SOLUCAO: usar .ravel() ou .flatten() para converter para 1D
#
# Mensagem de aviso original:
#   DataConversionWarning: A column-vector y was passed when a 1d array was expected
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

X, y_1d = load_iris(return_X_y=True)

# PROBLEMA: y como coluna (2D)
y_coluna = y_1d.reshape(-1, 1)
print("Shape incorreto:", y_coluna.shape)  # (150, 1)

# SOLUCAO 1: .ravel()
y_correto = y_coluna.ravel()
print("Shape correto (.ravel):", y_correto.shape)  # (150,)

# SOLUCAO 2: .flatten()
y_correto2 = y_coluna.flatten()
print("Shape correto (.flatten):", y_correto2.shape)  # (150,)

model = LogisticRegression(max_iter=200).fit(X, y_correto)
print(f"Modelo treinado! Acuracia: {model.score(X, y_correto):.2f}")
