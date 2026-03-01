# GO2114-35cLimeParaXai
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import lime
from lime import lime_tabular, lime_text
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("LIME - Explainability (XAI)")
print("="*70)

# ==================
# EXEMPLO 1: DADOS TABULARES (IRIS)
# ==================
print("\n📊 EXEMPLO 1: Explicando modelo tabular (Iris)\n")

# Carregar dados
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

accuracy = rf_clf.score(X_test, y_test)
print(f"Acurácia do modelo: {accuracy*100:.2f}%")

# Criar explainer LIME
explainer_tab = lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    mode='classification'
)

# Explicar uma predição
instance_idx = 0
instance = X_test[instance_idx]

print(f"\n🔍 Explicando instância {instance_idx}:")
print(f"  Features: {instance}")
print(f"  Classe real: {iris.target_names[y_test[instance_idx]]}")
print(f"  Classe predita: {iris.target_names[rf_clf.predict([instance])[0]]}")

# Gerar explicação
explanation = explainer_tab.explain_instance(
    instance,
    rf_clf.predict_proba,
    num_features=4
)

print("\n📋 Explicação (features mais importantes):")
for feature, weight in explanation.as_list():
    print(f"  {feature}: {weight:+.3f}")

# Visualizar
fig = explanation.as_pyplot_figure()
plt.tight_layout()
plt.savefig('lime_tabular_explanation.png', dpi=150)
print("\n✅ Explicação visual salva: lime_tabular_explanation.png")
plt.close()

# ==================
# EXEMPLO 2: DADOS DE TEXTO
# ==================
print("\n" + "="*70)
print("📝 EXEMPLO 2: Explicando classificação de texto\n")

# Carregar dados (subset para rapidez)
categories = ['alt.atheism', 'soc.religion.christian']
newsgroups_train = fetch_20newsgroups(
    subset='train',
    categories=categories,
    remove=('headers', 'footers', 'quotes')
)
newsgroups_test = fetch_20newsgroups(
    subset='test',
    categories=categories,
    remove=('headers', 'footers', 'quotes')
)

print(f"Documentos de treino: {len(newsgroups_train.data)}")
print(f"Documentos de teste: {len(newsgroups_test.data)}")

# Vetorizar texto
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_text = vectorizer.fit_transform(newsgroups_train.data)
X_test_text = vectorizer.transform(newsgroups_test.data)

y_train_text = newsgroups_train.target
y_test_text = newsgroups_test.target

# Treinar classificador
rf_text = RandomForestClassifier(n_estimators=100, random_state=42)
rf_text.fit(X_train_text, y_train_text)

accuracy_text = rf_text.score(X_test_text, y_test_text)
print(f"\nAcurácia do modelo: {accuracy_text*100:.2f}%")

# Pipeline de predição
class TextClassifier:
    def __init__(self, vectorizer, classifier):
        self.vectorizer = vectorizer
        self.classifier = classifier

    def predict_proba(self, texts):
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)

text_clf = TextClassifier(vectorizer, rf_text)

# Criar explainer LIME para texto
explainer_text = lime_text.LimeTextExplainer(class_names=newsgroups_test.target_names)

# Explicar uma predição
doc_idx = 10
doc = newsgroups_test.data[doc_idx]

print(f"\n🔍 Explicando documento {doc_idx}:")
print(f"\n--- Texto (primeiras 200 chars) ---")
print(doc[:200])
print("...")

prediction = rf_text.predict(vectorizer.transform([doc]))[0]
proba = rf_text.predict_proba(vectorizer.transform([doc]))[0]

print(f"\n📊 Predição:")
print(f"  Classe real: {newsgroups_test.target_names[y_test_text[doc_idx]]}")
print(f"  Classe predita: {newsgroups_test.target_names[prediction]}")
print(f"  Probabilidades: {proba}")

# Gerar explicação
explanation_text = explainer_text.explain_instance(
    doc,
    text_clf.predict_proba,
    num_features=10
)

print("\n📋 Palavras mais importantes:")
for word, weight in explanation_text.as_list():
    print(f"  '{word}': {weight:+.3f}")

# Visualizar no notebook (HTML)
html = explanation_text.as_html()
with open('lime_text_explanation.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("\n✅ Explicação HTML salva: lime_text_explanation.html")

# ==================
# EXEMPLO 3: ANÁLISE DE SENSIBILIDADE
# ==================
print("\n" + "="*70)
print("🔬 EXEMPLO 3: Análise de Sensibilidade\n")

# Testar como mudanças nas features afetam a predição
feature_idx = 0  # Petal length
feature_name = iris.feature_names[feature_idx]

original_value = instance[feature_idx]
values_range = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), 50)

predictions = []

for val in values_range:
    test_instance = instance.copy()
    test_instance[feature_idx] = val
    pred_proba = rf_clf.predict_proba([test_instance])[0]
    predictions.append(pred_proba)

predictions = np.array(predictions)

# Plotar
plt.figure(figsize=(12, 6))

for i, class_name in enumerate(iris.target_names):
    plt.plot(values_range, predictions[:, i], label=class_name, linewidth=2)

plt.axvline(original_value, color='red', linestyle='--', label='Valor Original')
plt.xlabel(feature_name, fontsize=12)
plt.ylabel('Probabilidade', fontsize=12)
plt.title(f'Sensibilidade da Predição a {feature_name}', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sensitivity_analysis.png', dpi=150)
print("✅ Análise de sensibilidade salva: sensitivity_analysis.png")

print("\n📊 RESUMO XAI COM LIME:")
print("  ✓ Explica QUALQUER modelo (model-agnostic)")
print("  ✓ Explicações locais (por instância)")
print("  ✓ Funciona com tabular, texto, imagem")
print("  ✓ Identifica features importantes")
print("  ✓ Ajuda a detectar bias e erros")

print("\n💡 OUTRAS TÉCNICAS XAI:")
print("  - SHAP: Teoria dos jogos cooperativos")
print("  - Grad-CAM: Heatmaps para CNNs")
print("  - Attention Weights: Transformers")
print("  - Partial Dependence Plots: ML clássico")
print("  - Counterfactual Explanations: 'What if?'")
