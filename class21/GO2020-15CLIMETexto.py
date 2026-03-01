# GO2020-15CLIMETexto
import lime
import lime.lime_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np

# ═══════════════════════════════════════════════════════════
# 1. DADOS SINTÉTICOS DE REVIEWS
# ═══════════════════════════════════════════════════════════
reviews = [
    "This movie is amazing! Great acting and plot.",  # Positivo
    "Terrible film. Waste of time and money.",        # Negativo
    "Loved every minute of it. Masterpiece!",         # Positivo
    "Boring and predictable. Don't recommend.",       # Negativo
    "Fantastic cinematography and soundtrack.",       # Positivo
    "Awful acting. Couldn't finish watching.",        # Negativo
    "One of the best movies I've ever seen!",         # Positivo
    "Poor script and weak characters.",               # Negativo
    "Brilliant performance by the cast.",             # Positivo
    "Disappointing ending. Not worth it.",            # Negativo
] * 50  # Repetir para ter mais dados

labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 50  # 1=positivo, 0=negativo

# ═══════════════════════════════════════════════════════════
# 2. TREINAR CLASSIFICADOR
# ═══════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    reviews, labels, test_size=0.2, random_state=42, stratify=labels
)

# Pipeline: TF-IDF + Logistic Regression
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=100, ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])

model.fit(X_train, y_train)
print(f"Acurácia: {model.score(X_test, y_test):.3f}")

# ═══════════════════════════════════════════════════════════
# 3. CRIAR EXPLAINER LIME
# ═══════════════════════════════════════════════════════════
explainer = lime.lime_text.LimeTextExplainer(
    class_names=['Negativo', 'Positivo'],
    random_state=42
)

# ═══════════════════════════════════════════════════════════
# 4. EXPLICAR PREDIÇÃO INDIVIDUAL
# ═══════════════════════════════════════════════════════════
test_review = "This movie is absolutely fantastic! Amazing acting and great story."
prediction = model.predict([test_review])[0]
probas = model.predict_proba([test_review])[0]

print(f"\n{'='*60}")
print(f"REVIEW: {test_review}")
print(f"{'='*60}")
print(f"Predição: {'Positivo' if prediction == 1 else 'Negativo'}")
print(f"Probabilidades: Negativo={probas[0]:.1%}, Positivo={probas[1]:.1%}")

# Gerar explicação LIME
exp = explainer.explain_instance(
    test_review,
    model.predict_proba,
    num_features=10,  # Top 10 palavras mais importantes
    num_samples=500   # Gerar 500 variações do texto
)

# Mostrar explicação
print(f"\nPalavras mais importantes:")
for word, weight in exp.as_list():
    direction = "→ POSITIVO" if weight > 0 else "→ NEGATIVO"
    print(f"  '{word}': {weight:+.3f} {direction}")

# Salvar visualização HTML
exp.save_to_file('lime_text_explanation.html')
print(f"\n✅ Visualização salva em: lime_text_explanation.html")

# ═══════════════════════════════════════════════════════════
# 5. TESTAR COM REVIEW NEGATIVO
# ═══════════════════════════════════════════════════════════
negative_review = "Terrible movie. Boring plot and awful acting. Waste of money."
prediction_neg = model.predict([negative_review])[0]
probas_neg = model.predict_proba([negative_review])[0]

print(f"\n{'='*60}")
print(f"REVIEW: {negative_review}")
print(f"{'='*60}")
print(f"Predição: {'Positivo' if prediction_neg == 1 else 'Negativo'}")
print(f"Probabilidades: Negativo={probas_neg[0]:.1%}, Positivo={probas_neg[1]:.1%}")

exp_neg = explainer.explain_instance(
    negative_review,
    model.predict_proba,
    num_features=10
)

print(f"\nPalavras mais importantes:")
for word, weight in exp_neg.as_list():
    direction = "→ POSITIVO" if weight > 0 else "→ NEGATIVO"
    print(f"  '{word}': {weight:+.3f} {direction}")

# ═══════════════════════════════════════════════════════════
# 6. ANÁLISE DE SENSIBILIDADE - O QUE MUDA A PREDIÇÃO?
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"ANÁLISE DE SENSIBILIDADE")
print(f"{'='*60}")

# Testar variações removendo palavras-chave
variations = [
    "This movie is fantastic! Amazing acting and great story.",  # Original
    "This movie is fantastic! acting and great story.",          # Sem "amazing"
    "This movie is fantastic! Amazing acting and story.",        # Sem "great"
    "This movie is fantastic! Amazing acting.",                  # Só acting
    "This movie is fantastic!",                                  # Só "fantastic"
    "This movie is acting and story.",                           # Sem palavras positivas
]

print(f"\nComo a predição muda ao remover palavras?\n")
for var in variations:
    pred = model.predict([var])[0]
    prob = model.predict_proba([var])[0][1]
    print(f"'{var}'")
    print(f"  → {'Positivo' if pred == 1 else 'Negativo'} ({prob:.1%})\n")
