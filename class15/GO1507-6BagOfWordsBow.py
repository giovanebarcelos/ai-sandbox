# GO1507-6BagOfWordsBow
from sklearn.feature_extraction.text import TfidfVectorizer


if __name__ == "__main__":
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(textos)
    print(X.toarray())
