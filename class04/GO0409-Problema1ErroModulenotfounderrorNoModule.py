# GO0409 - Problema 1: ModuleNotFoundError -- No module named sklearn
# ERRO COMUM: tentar importar biblioteca nao instalada
# SOLUCAO: instalar com pip ou conda
#
# Mensagem de erro original:
#   ModuleNotFoundError: No module named 'sklearn'
#
# SOLUCAO:
#   pip install scikit-learn
#   ou: conda install scikit-learn
import sys

print("Verificando instalacao do scikit-learn...")
try:
    import sklearn
    print(f"OK: sklearn versao {sklearn.__version__}")
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200).fit(X, y)
    print(f"Modelo treinado, acuracia: {model.score(X, y):.2f}")
except ImportError:
    print("ERRO: sklearn nao instalado.")
    print("Execute: pip install scikit-learn")
    sys.exit(1)
