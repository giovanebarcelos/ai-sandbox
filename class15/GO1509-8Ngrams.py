# GO1509-8Ngrams
from sklearn.feature_extraction.text import CountVectorizer


if __name__ == "__main__":
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    X = vectorizer.fit_transform(textos)
    print(vectorizer.get_feature_names_out())

    vectorizer = CountVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(textos)
