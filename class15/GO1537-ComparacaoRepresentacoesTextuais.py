# GO1537-ComparacaoRepresentacoesTextuais
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')

# Carregar dados preprocessados
df_train = pd.read_csv('imdb_train_preprocessed.csv')
df_test = pd.read_csv('imdb_test_preprocessed.csv')

X_train_text = df_train['text_clean']
y_train = df_train['sentiment']
X_test_text = df_test['text_clean']
y_test = df_test['sentiment']

print("="*80)
print("COMPARAÇÃO DE REPRESENTAÇÕES TEXTUAIS")
print("="*80)

# ============================================================================
# MÉTODO 1: BAG-OF-WORDS (BoW)
# ============================================================================
print("\n1️⃣ BAG-OF-WORDS (CountVectorizer)")
print("-" * 80)

# Criar vetorizador BoW
bow_vectorizer = CountVectorizer(
    max_features=5000,  # Top 5000 palavras mais frequentes
    ngram_range=(1, 2),  # Unigrams + bigrams
    min_df=5  # Ignorar palavras que aparecem menos de 5 vezes
)

# Transformar textos em vetores
X_train_bow = bow_vectorizer.fit_transform(X_train_text)
X_test_bow = bow_vectorizer.transform(X_test_text)

print(f"Forma da matriz BoW (treino): {X_train_bow.shape}")
print(f"Forma da matriz BoW (teste): {X_test_bow.shape}")
print(f"Vocabulário: {len(bow_vectorizer.vocabulary_)} palavras")
print(f"Esparsidade: {(1.0 - X_train_bow.nnz / (X_train_bow.shape[0] * X_train_bow.shape[1])) * 100:.2f}%")

# Treinar classificador
model_bow = LogisticRegression(max_iter=1000, random_state=42)
model_bow.fit(X_train_bow, y_train)

# Avaliar
y_pred_bow = model_bow.predict(X_test_bow)
acc_bow = accuracy_score(y_test, y_pred_bow)
print(f"\n✅ Acurácia BoW: {acc_bow:.4f}")

# ============================================================================
# MÉTODO 2: TF-IDF
# ============================================================================
print("\n2️⃣ TF-IDF (TfidfVectorizer)")
print("-" * 80)

# Criar vetorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=5,
    sublinear_tf=True  # Aplicar escala logarítmica
)

# Transformar textos
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

print(f"Forma da matriz TF-IDF (treino): {X_train_tfidf.shape}")
print(f"Forma da matriz TF-IDF (teste): {X_test_tfidf.shape}")

# Treinar classificador
model_tfidf = LogisticRegression(max_iter=1000, random_state=42)
model_tfidf.fit(X_train_tfidf, y_train)

# Avaliar
y_pred_tfidf = model_tfidf.predict(X_test_tfidf)
acc_tfidf = accuracy_score(y_test, y_pred_tfidf)
print(f"\n✅ Acurácia TF-IDF: {acc_tfidf:.4f}")

# ============================================================================
# MÉTODO 3: WORD2VEC EMBEDDINGS
# ============================================================================
print("\n3️⃣ WORD2VEC EMBEDDINGS")
print("-" * 80)

# Preparar corpus para Word2Vec (lista de listas de tokens)
train_sentences = [text.split() for text in X_train_text]

# Treinar modelo Word2Vec
print("Treinando Word2Vec (isso pode demorar alguns minutos)...")
w2v_model = Word2Vec(
    sentences=train_sentences,
    vector_size=100,  # Dimensão dos embeddings
    window=5,  # Contexto de ±5 palavras
    min_count=5,  # Ignorar palavras raras
    workers=4,  # Paralelização
    sg=1,  # Skip-gram (1) vs CBOW (0)
    epochs=10
)

print(f"Vocabulário Word2Vec: {len(w2v_model.wv)} palavras")
print(f"Dimensão dos vetores: {w2v_model.wv.vector_size}")

# Função para calcular embedding médio de um documento
def document_vector(text, model):
    """Calcula o vetor médio de um documento"""
    words = text.split()
    # Filtrar palavras que estão no vocabulário
    word_vectors = [model.wv[word] for word in words if word in model.wv]

    if len(word_vectors) == 0:
        # Se nenhuma palavra está no vocabulário, retornar vetor zero
        return np.zeros(model.wv.vector_size)

    # Retornar média dos vetores
    return np.mean(word_vectors, axis=0)

# Transformar documentos em vetores
print("Transformando documentos em embeddings...")
X_train_w2v = np.array([document_vector(text, w2v_model) for text in X_train_text])
X_test_w2v = np.array([document_vector(text, w2v_model) for text in X_test_text])

print(f"Forma da matriz Word2Vec (treino): {X_train_w2v.shape}")
print(f"Forma da matriz Word2Vec (teste): {X_test_w2v.shape}")

# Treinar classificador
model_w2v = LogisticRegression(max_iter=1000, random_state=42)
model_w2v.fit(X_train_w2v, y_train)

# Avaliar
y_pred_w2v = model_w2v.predict(X_test_w2v)
acc_w2v = accuracy_score(y_test, y_pred_w2v)
print(f"\n✅ Acurácia Word2Vec: {acc_w2v:.4f}")

# ============================================================================
# COMPARAÇÃO VISUAL DOS RESULTADOS
# ============================================================================
print("\n" + "="*80)
print("COMPARAÇÃO FINAL")
print("="*80)

# Criar DataFrame de comparação
comparison_df = pd.DataFrame({
    'Método': ['Bag-of-Words', 'TF-IDF', 'Word2Vec'],
    'Acurácia': [acc_bow, acc_tfidf, acc_w2v],
    'Features': [X_train_bow.shape[1], X_train_tfidf.shape[1], X_train_w2v.shape[1]]
})

print(comparison_df.to_string(index=False))

# Visualização
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Comparação de acurácias
axes[0].bar(comparison_df['Método'], comparison_df['Acurácia'], 
            color=['#3498db', '#2ecc71', '#e74c3c'])
axes[0].set_ylabel('Acurácia')
axes[0].set_title('Comparação de Acurácias')
axes[0].set_ylim([0.8, 0.95])
axes[0].axhline(y=0.5, color='gray', linestyle='--', label='Baseline (50%)')
for i, v in enumerate(comparison_df['Acurácia']):
    axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')

# Gráfico 2: Número de features
axes[1].bar(comparison_df['Método'], comparison_df['Features'], 
            color=['#3498db', '#2ecc71', '#e74c3c'])
axes[1].set_ylabel('Número de Features')
axes[1].set_title('Dimensionalidade das Representações')
axes[1].set_yscale('log')
for i, v in enumerate(comparison_df['Features']):
    axes[1].text(i, v * 1.2, f'{v}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('aula15_comparacao_representacoes.png', dpi=300, bbox_inches='tight')
print("\n📊 Gráfico salvo: aula15_comparacao_representacoes.png")

# Relatórios detalhados
print("\n" + "="*80)
print("RELATÓRIOS DETALHADOS")
print("="*80)

for method_name, y_pred in [('BoW', y_pred_bow), ('TF-IDF', y_pred_tfidf), ('Word2Vec', y_pred_w2v)]:
    print(f"\n{method_name}:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Negativo', 'Positivo']))

# Análise de palavras importantes (TF-IDF)
print("\n" + "="*80)
print("TOP 10 PALAVRAS MAIS IMPORTANTES (TF-IDF)")
print("="*80)

feature_names = tfidf_vectorizer.get_feature_names_out()
coef = model_tfidf.coef_[0]

# Palavras mais positivas
top_positive_idx = np.argsort(coef)[-10:]
print("\nPalavras mais POSITIVAS:")
for idx in reversed(top_positive_idx):
    print(f"  {feature_names[idx]:20s} → {coef[idx]:.4f}")

# Palavras mais negativas
top_negative_idx = np.argsort(coef)[:10]
print("\nPalavras mais NEGATIVAS:")
for idx in top_negative_idx:
    print(f"  {feature_names[idx]:20s} → {coef[idx]:.4f}")

# Testar similaridade com Word2Vec
print("\n" + "="*80)
print("EXEMPLOS DE SIMILARIDADE (Word2Vec)")
print("="*80)

test_words = ['good', 'bad', 'excellent', 'terrible', 'boring', 'exciting']
for word in test_words:
    if word in w2v_model.wv:
        similar = w2v_model.wv.most_similar(word, topn=5)
        print(f"\n'{word}' é similar a:")
        for sim_word, score in similar:
            print(f"  {sim_word:15s} (score: {score:.4f})")

print("\n✅ Comparação completa realizada!")
