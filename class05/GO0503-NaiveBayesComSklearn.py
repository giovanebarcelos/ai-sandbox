# GO0503-NaiveBayesComSklearn
# ═══════════════════════════════════════════════════════════════════
# NAIVE BAYES COM SKLEARN
# ═══════════════════════════════════════════════════════════════════

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# ───────────────────────────────────────────────────────────────────
# EXEMPLO 1: GAUSSIAN NB (dados numéricos)
# ───────────────────────────────────────────────────────────────────

print("="*60)
print("GAUSSIAN NAIVE BAYES - Iris Dataset")
print("="*60)

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

print(f"Acurácia: {gnb.score(X_test, y_test):.3f}")

# Predição com probabilidades
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = gnb.predict(sample)
probabilities = gnb.predict_proba(sample)[0]

print(f"\nPredição para {sample}:")
print(f"Classe predita: {iris.target_names[prediction[0]]}")
for i, prob in enumerate(probabilities):
    print(f"  P({iris.target_names[i]}): {prob:.3f}")

# ───────────────────────────────────────────────────────────────────
# EXEMPLO 2: MULTINOMIAL NB (classificação de textos)
# ───────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("MULTINOMIAL NAIVE BAYES - Text Classification")
print("="*60)

# Carregar dataset de textos (20newsgroups - versão pequena)
categories = ['alt.atheism', 'soc.religion.christian', 
              'comp.graphics', 'sci.med']

train_data = fetch_20newsgroups(subset='train', 
                                 categories=categories,
                                 shuffle=True,
                                 random_state=42)

# Converter texto para features numéricas (bag-of-words)
vectorizer = CountVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(train_data.data)

# Treinar Multinomial NB
mnb = MultinomialNB(alpha=1.0)  # alpha = Laplace smoothing
mnb.fit(X_train_vec, train_data.target)

# Testar
test_data = fetch_20newsgroups(subset='test', 
                                categories=categories,
                                shuffle=True,
                                random_state=42)

X_test_vec = vectorizer.transform(test_data.data)
accuracy = mnb.score(X_test_vec, test_data.target)

print(f"Acurácia: {accuracy:.3f}")

# Classificar um texto novo
new_text = ["God is love", "Medical treatment for cancer"]
new_vec = vectorizer.transform(new_text)
predictions = mnb.predict(new_vec)

for text, pred in zip(new_text, predictions):
    print(f"\n'{text}'")
    print(f"  → Categoria: {train_data.target_names[pred]}")
