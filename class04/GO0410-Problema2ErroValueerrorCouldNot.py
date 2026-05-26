# GO0410 - Problema 2: ValueError -- could not convert string to float
# ERRO COMUM: passar coluna de texto diretamente para modelo sklearn
# SOLUCAO: usar LabelEncoder ou pd.get_dummies para codificar
#
# Mensagem de erro original:
#   ValueError: could not convert string to float: 'categoria_texto'
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Dados com categoria textual (problema)
categorias = ['gato', 'cachorro', 'gato', 'passaro', 'cachorro']
y = [0, 1, 0, 2, 1]

# SOLUCAO: LabelEncoder converte texto para inteiro
le = LabelEncoder()
X_encoded = le.fit_transform(categorias).reshape(-1, 1)
print("Categorias originais:", categorias)
print("Encoded:", X_encoded.ravel())
print("Classes:", le.classes_)

# Agora sklearn aceita
model = LogisticRegression().fit(X_encoded, y)
print("Modelo treinado com sucesso!")
