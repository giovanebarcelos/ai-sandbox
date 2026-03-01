# GO1114-TreinamentoHibrido
# pip install anfis numpy
from anfis import ANFIS
import numpy as np

# Dados de treino


if __name__ == "__main__":
    X_train = np.array([[...], [...]])
    y_train = np.array([...])

    # Definir funções de pertinência
    mf = [[['gaussmf', {'mean': 0., 'sigma': 1.}],
           ['gaussmf', {'mean': 1., 'sigma': 2.}]],
          [['gaussmf', {'mean': 0., 'sigma': 1.}],
           ['gaussmf', {'mean': 1., 'sigma': 2.}]]]

    # Criar ANFIS
    anf = ANFIS(X_train, y_train, mf)

    # Treinar
    anf.train(epochs=100)

    # Prever
    y_pred = anf.predict(X_test)
