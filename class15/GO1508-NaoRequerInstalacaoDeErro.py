# GO1508-NãoRequerInstalaçãoDeErro
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class TFIDFAnalyzer:
    """
    Advanced TF-IDF analysis and visualization

    Features:
    - Compute TF-IDF scores
    - Extract top keywords per document
    - Visualize feature importance
    - Compare documents by TF-IDF similarity
    """

    def __init__(self, max_features=None, min_df=1, max_df=1.0):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words='english'
        )
        self.tfidf_matrix = None
        self.feature_names = None

    def fit_transform(self, documents):
        """Fit and transform documents"""
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()

        print(f"✅ TF-IDF Matrix created:")
        print(f"   Documents: {self.tfidf_matrix.shape[0]}")
        print(f"   Features (vocabulary): {self.tfidf_matrix.shape[1]}")
        print(f"   Sparsity: {(1 - self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1]))*100:.2f}%\n")

        return self.tfidf_matrix

    def get_top_keywords(self, doc_idx, top_n=10):
        """Get top keywords for a document"""
        if self.tfidf_matrix is None:
            raise ValueError("Call fit_transform first!")

        # Get TF-IDF scores for the document
        doc_vector = self.tfidf_matrix[doc_idx].toarray()[0]

        # Get top indices
        top_indices = doc_vector.argsort()[-top_n:][::-1]

        # Get words and scores
        top_words = [(self.feature_names[i], doc_vector[i]) for i in top_indices if doc_vector[i] > 0]

        return top_words

    def compute_similarity(self, doc_idx1, doc_idx2):
        """Compute cosine similarity between two documents"""
        from sklearn.metrics.pairwise import cosine_similarity

        vec1 = self.tfidf_matrix[doc_idx1]
        vec2 = self.tfidf_matrix[doc_idx2]

        return cosine_similarity(vec1, vec2)[0][0]

    def get_similarity_matrix(self):
        """Compute pairwise similarity matrix"""
        from sklearn.metrics.pairwise import cosine_similarity

        sim_matrix = cosine_similarity(self.tfidf_matrix)
        return sim_matrix

    def analyze_vocabulary(self):
        """Analyze vocabulary statistics"""
        idf_values = self.vectorizer.idf_

        vocab_stats = {
            'word': self.feature_names,
            'idf': idf_values
        }

        df = pd.DataFrame(vocab_stats)
        df = df.sort_values('idf', ascending=False)

        return df

# === DEMO ===

print("📊 Advanced TF-IDF Analysis\n")
print("="*70)

# Sample documents
documents = [
    "Machine learning algorithms learn patterns from data automatically",
    "Deep learning uses neural networks with multiple layers",
    "Natural language processing enables computers to understand text",
    "Computer vision allows machines to interpret visual information",
    "Data science combines statistics programming and domain knowledge",
    "Artificial intelligence transforms how we solve complex problems",
]

print("📚 Corpus: {} documents\n".format(len(documents)))

# Initialize analyzer
analyzer = TFIDFAnalyzer(max_features=50)

# Fit and transform
tfidf_matrix = analyzer.fit_transform(documents)

print("="*70)
print("\n🔑 TOP KEYWORDS PER DOCUMENT:\n")

for i, doc in enumerate(documents):
    print(f"Document {i+1}: \"{doc[:50]}...\"")
    top_keywords = analyzer.get_top_keywords(i, top_n=5)
    for word, score in top_keywords:
        print(f"   {word}: {score:.4f}")
    print()

# Similarity analysis
print("="*70)
print("\n🔍 DOCUMENT SIMILARITY:\n")

sim_matrix = analyzer.get_similarity_matrix()

for i in range(min(3, len(documents))):
    for j in range(i+1, min(4, len(documents))):
        sim = analyzer.compute_similarity(i, j)
        print(f"Doc {i+1} vs Doc {j+1}: {sim:.4f}")

# Vocabulary analysis
print("\n" + "="*70)
print("\n📚 VOCABULARY ANALYSIS:\n")

vocab_df = analyzer.analyze_vocabulary()

print("Top 10 most distinctive words (highest IDF):\n")
for _, row in vocab_df.head(10).iterrows():
    print(f"   {row['word']}: {row['idf']:.4f}")

print("\nTop 10 most common words (lowest IDF):\n")
for _, row in vocab_df.tail(10).iterrows():
    print(f"   {row['word']}: {row['idf']:.4f}")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. TF-IDF heatmap (documents x top features)
ax = axes[0, 0]

top_features = 15
top_feature_indices = vocab_df.head(top_features).index
top_feature_names = [analyzer.feature_names[i] for i in range(min(top_features, len(analyzer.feature_names)))]

tfidf_array = tfidf_matrix.toarray()[:, :top_features]

sns.heatmap(tfidf_array, cmap='YlOrRd', ax=ax,
            xticklabels=top_feature_names[:tfidf_array.shape[1]],
            yticklabels=[f'Doc {i+1}' for i in range(len(documents))],
            cbar_kws={'label': 'TF-IDF Score'})
ax.set_title('TF-IDF Scores Heatmap')
ax.set_xlabel('Features')
ax.set_ylabel('Documents')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# 2. Document similarity heatmap
ax = axes[0, 1]

sns.heatmap(sim_matrix, cmap='RdYlGn', vmin=0, vmax=1, ax=ax,
            xticklabels=[f'D{i+1}' for i in range(len(documents))],
            yticklabels=[f'D{i+1}' for i in range(len(documents))],
            annot=True, fmt='.2f', cbar_kws={'label': 'Cosine Similarity'})
ax.set_title('Document Similarity Matrix')

# 3. IDF distribution
ax = axes[1, 0]

idf_values = vocab_df['idf'].values
ax.hist(idf_values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
ax.axvline(idf_values.mean(), color='red', linestyle='--', linewidth=2,
          label=f'Mean: {idf_values.mean():.2f}')
ax.set_xlabel('IDF Value')
ax.set_ylabel('Frequency')
ax.set_title('IDF Score Distribution')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 4. Top keywords bar chart
ax = axes[1, 1]

doc_to_visualize = 0
top_keywords = analyzer.get_top_keywords(doc_to_visualize, top_n=10)

if top_keywords:
    words, scores = zip(*top_keywords)
    ax.barh(words, scores, color='coral', alpha=0.7)
    ax.set_xlabel('TF-IDF Score')
    ax.set_title(f'Top 10 Keywords in Document {doc_to_visualize+1}')
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig('tfidf_advanced_analysis.png', dpi=150, bbox_inches='tight')
print("\n📊 Gráfico salvo: tfidf_advanced_analysis.png")

print("\n✅ TF-IDF analysis completo!")
print("\n💡 KEY CONCEPTS:")
print("   - TF (Term Frequency): How often word appears in document")
print("   - IDF (Inverse Document Frequency): Rarity across corpus")
print("   - TF-IDF = TF × IDF: Balance frequency with distinctiveness")
print("   - High TF-IDF = important & distinctive word for document")
print("\n💡 PARAMETERS:")
print("   - max_features: Limit vocabulary size")
print("   - min_df: Ignore terms appearing in < min_df documents")
print("   - max_df: Ignore terms appearing in > max_df% documents")
print("   - ngram_range: Include n-grams (1,2) for unigrams + bigrams")
