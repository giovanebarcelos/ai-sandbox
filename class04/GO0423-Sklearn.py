# GO0423-Sklearn
# Abordagem ML - aprende padrões automaticamente
from sklearn.naive_bayes import MultinomialNB

# 1. Treinar com exemplos


if __name__ == "__main__":
    modelo = MultinomialNB()
    modelo.fit(emails_treino, labels_treino)  # 10.000 emails rotulados

    # 2. Usar o modelo treinado
    predicao = modelo.predict([novo_email])  # Detecta spam automaticamente
