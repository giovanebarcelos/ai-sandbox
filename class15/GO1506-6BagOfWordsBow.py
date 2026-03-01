# GO1506-6BagOfWordsBow
from sklearn.feature_extraction.text import CountVectorizer


if __name__ == "__main__":
    textos = [
        "Eu gosto de Python",
        "Python é ótimo",
        "Eu amo programação"
    ]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(textos)

    print(vectorizer.get_feature_names_out())

    print(X.toarray())
