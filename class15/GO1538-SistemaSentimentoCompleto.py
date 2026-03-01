# GO1538-SistemaSentimentoCompleto
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ============================================================================
# CLASSE DO SISTEMA DE ANÁLISE DE SENTIMENTO
# ============================================================================

class SentimentAnalyzer:
    """
    Sistema completo de análise de sentimento
    """
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, text):
        """Preprocessa um texto"""
        # Converter para minúsculas
        text = text.lower()
        # Remover HTML e URLs
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remover números e caracteres especiais
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        # Tokenizar
        tokens = word_tokenize(text)
        # Remover stopwords e lematizar
        tokens = [self.lemmatizer.lemmatize(word) 
                  for word in tokens 
                  if word not in self.stop_words and len(word) > 2]
        return ' '.join(tokens)

    def train(self, X_train, y_train, max_features=5000):
        """Treina o modelo de análise de sentimento"""
        print("🔨 Treinando modelo de análise de sentimento...")

        # Preprocessar textos
        print("  → Preprocessando textos...")
        X_train_clean = [self.preprocess(text) for text in X_train]

        # Criar vetorizador TF-IDF
        print("  → Criando vetorização TF-IDF...")
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=5,
            sublinear_tf=True
        )

        X_train_tfidf = self.vectorizer.fit_transform(X_train_clean)

        # Treinar modelo
        print("  → Treinando classificador...")
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X_train_tfidf, y_train)

        print("✅ Treinamento concluído!")
        return self

    def predict(self, text):
        """Prediz o sentimento de um texto"""
        # Preprocessar
        text_clean = self.preprocess(text)
        # Vetorizar
        text_vector = self.vectorizer.transform([text_clean])
        # Predizer
        prediction = self.model.predict(text_vector)[0]
        probability = self.model.predict_proba(text_vector)[0]

        return {
            'sentiment': 'Positivo' if prediction == 1 else 'Negativo',
            'confidence': max(probability),
            'probabilities': {
                'negativo': probability[0],
                'positivo': probability[1]
            }
        }

    def predict_batch(self, texts):
        """Prediz o sentimento de múltiplos textos"""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results

    def evaluate(self, X_test, y_test):
        """Avalia o modelo no conjunto de teste"""
        print("📊 Avaliando modelo...")

        # Preprocessar
        X_test_clean = [self.preprocess(text) for text in X_test]

        # Vetorizar
        X_test_vector = self.vectorizer.transform(X_test_clean)

        # Predizer
        y_pred = self.model.predict(X_test_vector)

        # Métricas
        print("\n" + "="*80)
        print("RELATÓRIO DE CLASSIFICAÇÃO")
        print("="*80)
        print(classification_report(y_test, y_pred, 
                                    target_names=['Negativo', 'Positivo']))

        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negativo', 'Positivo'],
                    yticklabels=['Negativo', 'Positivo'])
        plt.title('Matriz de Confusão - Análise de Sentimento')
        plt.ylabel('Real')
        plt.xlabel('Predito')
        plt.savefig('aula15_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\n📊 Matriz de confusão salva: aula15_confusion_matrix.png")

        return y_pred

    def save(self, filepath='sentiment_analyzer.pkl'):
        """Salva o modelo treinado"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"💾 Modelo salvo: {filepath}")

    @staticmethod
    def load(filepath='sentiment_analyzer.pkl'):
        """Carrega um modelo salvo"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def get_important_words(self, n=20):
        """Retorna as palavras mais importantes para classificação"""
        feature_names = self.vectorizer.get_feature_names_out()
        coef = self.model.coef_[0]

        # Palavras positivas
        top_positive_idx = np.argsort(coef)[-n:]
        positive_words = [(feature_names[idx], coef[idx]) 
                          for idx in reversed(top_positive_idx)]

        # Palavras negativas
        top_negative_idx = np.argsort(coef)[:n]
        negative_words = [(feature_names[idx], coef[idx]) 
                          for idx in top_negative_idx]

        return {
            'positive': positive_words,
            'negative': negative_words
        }

# ============================================================================
# USAR O SISTEMA
# ============================================================================

# Carregar dados
df_train = pd.read_csv('imdb_train_preprocessed.csv')
df_test = pd.read_csv('imdb_test_preprocessed.csv')

# Criar e treinar o analisador
analyzer = SentimentAnalyzer()
analyzer.train(df_train['text'], df_train['sentiment'])

# Avaliar no conjunto de teste
analyzer.evaluate(df_test['text'], df_test['sentiment'])

# Salvar modelo
analyzer.save('sentiment_analyzer_imdb.pkl')

# ============================================================================
# TESTAR COM NOVOS REVIEWS
# ============================================================================

print("\n" + "="*80)
print("TESTANDO COM NOVOS REVIEWS")
print("="*80)

new_reviews = [
    "This movie was absolutely fantastic! Best film I've seen this year!",
    "Terrible waste of time. The plot was nonsense and acting was awful.",
    "It was okay, nothing special but not terrible either.",
    "Brilliant performances and stunning cinematography. A masterpiece!",
    "I fell asleep halfway through. Boring and predictable.",
    "The special effects were amazing but the story was weak."
]

for i, review in enumerate(new_reviews, 1):
    result = analyzer.predict(review)
    print(f"\n{i}. Review: {review[:60]}...")
    print(f"   Sentimento: {result['sentiment']}")
    print(f"   Confiança: {result['confidence']:.2%}")
    print(f"   Prob. Negativo: {result['probabilities']['negativo']:.2%}")
    print(f"   Prob. Positivo: {result['probabilities']['positivo']:.2%}")

# ============================================================================
# ANÁLISE DE PALAVRAS IMPORTANTES
# ============================================================================

print("\n" + "="*80)
print("PALAVRAS MAIS IMPORTANTES PARA CLASSIFICAÇÃO")
print("="*80)

important_words = analyzer.get_important_words(n=15)

print("\n🟢 TOP 15 PALAVRAS POSITIVAS:")
for word, score in important_words['positive']:
    print(f"  {word:20s} → {score:.4f}")

print("\n🔴 TOP 15 PALAVRAS NEGATIVAS:")
for word, score in important_words['negative']:
    print(f"  {word:20s} → {score:.4f}")

# Visualizar palavras importantes
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Palavras positivas
pos_words, pos_scores = zip(*important_words['positive'][:10])
axes[0].barh(range(len(pos_words)), pos_scores, color='green', alpha=0.7)
axes[0].set_yticks(range(len(pos_words)))
axes[0].set_yticklabels(pos_words)
axes[0].set_xlabel('Peso do Coeficiente')
axes[0].set_title('Top 10 Palavras Positivas')
axes[0].invert_yaxis()

# Palavras negativas
neg_words, neg_scores = zip(*important_words['negative'][:10])
axes[1].barh(range(len(neg_words)), [abs(s) for s in neg_scores], 
             color='red', alpha=0.7)
axes[1].set_yticks(range(len(neg_words)))
axes[1].set_yticklabels(neg_words)
axes[1].set_xlabel('Peso do Coeficiente (valor absoluto)')
axes[1].set_title('Top 10 Palavras Negativas')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('aula15_important_words.png', dpi=300, bbox_inches='tight')
print("\n📊 Gráfico salvo: aula15_important_words.png")

# ============================================================================
# INTERFACE INTERATIVA SIMPLES
# ============================================================================

def analyze_sentiment_interactive():
    """Interface interativa para análise de sentimento"""
    print("\n" + "="*80)
    print("SISTEMA DE ANÁLISE DE SENTIMENTO - MODO INTERATIVO")
    print("="*80)
    print("Digite 'sair' para encerrar")

    # Carregar modelo
    analyzer = SentimentAnalyzer.load('sentiment_analyzer_imdb.pkl')

    while True:
        print("\n" + "-"*80)
        text = input("Digite um review de filme: ")

        if text.lower() in ['sair', 'exit', 'quit']:
            print("Encerrando...")
            break

        if not text.strip():
            continue

        result = analyzer.predict(text)

        print("\n📊 RESULTADO:")
        print(f"  Sentimento: {result['sentiment']}")
        print(f"  Confiança: {result['confidence']:.2%}")

        # Barra visual de confiança
        bar_length = 40
        pos_bars = int(result['probabilities']['positivo'] * bar_length)
        neg_bars = bar_length - pos_bars

        print(f"\n  [-] {'█' * neg_bars}{' ' * pos_bars} [+]")
        print(f"      {result['probabilities']['negativo']:.1%}{'':>30s}{result['probabilities']['positivo']:.1%}")

# Descomentar para usar modo interativo:
# analyze_sentiment_interactive()

print("\n✅ Sistema de análise de sentimento completo implementado!")
print("💡 Use: analyzer.predict('seu texto aqui') para analisar novos reviews")
