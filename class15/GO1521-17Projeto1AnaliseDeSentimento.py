# GO1521-17Projeto1AnáliseDeSentimento
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import re
import string


if __name__ == "__main__":
    df = pd.read_csv('tweets_sentiment.csv')

    def clean_tweet(text):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    df['text_clean'] = df['text'].apply(clean_tweet)

    X_train, X_test, y_train, y_test = train_test_split(
        df['text_clean'], df['sentiment'],
        test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print(classification_report(y_test, y_pred))

    def predict_sentiment(tweet):
        clean = clean_tweet(tweet)
        vec = vectorizer.transform([clean])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        return "Positivo" if pred == 1 else "Negativo", prob

    tweet = "Adorei o novo produto! Muito bom!"
    sentimento, prob = predict_sentiment(tweet)
    print(f'{sentimento} (confiança: {prob.max():.2f})')
