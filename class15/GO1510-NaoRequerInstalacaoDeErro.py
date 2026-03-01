# GO1510-NãoRequerInstalaçãoDeErro
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import numpy as np

class NgramAnalyzer:
    """
    Analyze n-grams in text corpus

    N-gram types:
    - Unigram (1): Single words
    - Bigram (2): Word pairs
    - Trigram (3): Word triples
    - Character n-grams: For morphology
    """

    def __init__(self, ngram_range=(1, 1), max_features=None):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            stop_words='english'
        )
        self.ngram_counts = None
        self.feature_names = None

    def fit_transform(self, documents):
        """Extract n-grams from documents"""
        matrix = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()

        # Sum across documents
        self.ngram_counts = np.asarray(matrix.sum(axis=0)).ravel()

        print(f"✅ N-gram extraction complete:")
        print(f"   N-gram range: {self.ngram_range}")
        print(f"   Total n-grams: {len(self.feature_names)}")
        print(f"   Documents: {len(documents)}\n")

        return matrix

    def get_top_ngrams(self, top_n=20):
        """Get most frequent n-grams"""
        if self.ngram_counts is None:
            raise ValueError("Call fit_transform first!")

        # Sort by frequency
        top_indices = self.ngram_counts.argsort()[-top_n:][::-1]

        top_ngrams = [
            (self.feature_names[i], self.ngram_counts[i])
            for i in top_indices
        ]

        return top_ngrams

    def compare_ngram_ranges(self, documents, ranges=[(1,1), (2,2), (3,3)]):
        """Compare different n-gram ranges"""
        results = {}

        for ngram_range in ranges:
            vectorizer = CountVectorizer(
                ngram_range=ngram_range,
                max_features=self.max_features,
                stop_words='english'
            )

            matrix = vectorizer.fit_transform(documents)
            feature_names = vectorizer.get_feature_names_out()
            ngram_counts = np.asarray(matrix.sum(axis=0)).ravel()

            # Get top 10
            top_indices = ngram_counts.argsort()[-10:][::-1]
            top_ngrams = [
                (feature_names[i], ngram_counts[i])
                for i in top_indices
            ]

            results[ngram_range] = {
                'count': len(feature_names),
                'top': top_ngrams
            }

        return results

# === DEMO ===

print("🔢 N-gram Analysis Demo\n")
print("="*70)

# Sample corpus
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks for complex patterns",
    "Natural language processing enables text understanding",
    "Data science combines statistics and programming skills",
    "Python programming language is popular for data analysis",
    "Artificial intelligence transforms modern technology applications",
    "Neural networks learn from large datasets automatically",
    "Machine learning algorithms improve with more training data",
] * 2  # Repeat for better statistics

print(f"📚 Corpus: {len(documents)} documents\n")

# Analyze unigrams
print("="*70)
print("\n1️⃣ UNIGRAM ANALYSIS (single words)\n")

unigram_analyzer = NgramAnalyzer(ngram_range=(1, 1), max_features=100)
unigram_analyzer.fit_transform(documents)

top_unigrams = unigram_analyzer.get_top_ngrams(top_n=10)

print("Top 10 Unigrams:\n")
for ngram, count in top_unigrams:
    print(f"   {ngram}: {count}")

# Analyze bigrams
print("\n" + "="*70)
print("\n2️⃣ BIGRAM ANALYSIS (word pairs)\n")

bigram_analyzer = NgramAnalyzer(ngram_range=(2, 2), max_features=100)
bigram_analyzer.fit_transform(documents)

top_bigrams = bigram_analyzer.get_top_ngrams(top_n=10)

print("Top 10 Bigrams:\n")
for ngram, count in top_bigrams:
    print(f"   '{ngram}': {count}")

# Analyze trigrams
print("\n" + "="*70)
print("\n3️⃣ TRIGRAM ANALYSIS (word triples)\n")

trigram_analyzer = NgramAnalyzer(ngram_range=(3, 3), max_features=100)
trigram_analyzer.fit_transform(documents)

top_trigrams = trigram_analyzer.get_top_ngrams(top_n=10)

print("Top 10 Trigrams:\n")
for ngram, count in top_trigrams:
    print(f"   '{ngram}': {count}")

# Compare all ranges
print("\n" + "="*70)
print("\n📊 COMPARISON ACROSS N-GRAM RANGES\n")

comparison = unigram_analyzer.compare_ngram_ranges(
    documents,
    ranges=[(1,1), (2,2), (3,3)]
)

for ngram_range, stats in comparison.items():
    n = ngram_range[0] if ngram_range[0] == ngram_range[1] else f"{ngram_range[0]}-{ngram_range[1]}"
    print(f"N={n}: {stats['count']} unique n-grams")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Top unigrams
ax = axes[0, 0]
words, counts = zip(*top_unigrams[:10])
ax.barh(words, counts, color='skyblue', alpha=0.7)
ax.set_xlabel('Frequency')
ax.set_title('Top 10 Unigrams')
ax.invert_yaxis()

# 2. Top bigrams
ax = axes[0, 1]
bigrams, counts = zip(*top_bigrams[:8])
bigrams_short = [b[:20] + '...' if len(b) > 20 else b for b in bigrams]
ax.barh(bigrams_short, counts, color='lightgreen', alpha=0.7)
ax.set_xlabel('Frequency')
ax.set_title('Top 8 Bigrams')
ax.invert_yaxis()

# 3. N-gram vocabulary size comparison
ax = axes[1, 0]
ngram_types = ['Unigrams', 'Bigrams', 'Trigrams']
vocab_sizes = [comparison[(1,1)]['count'], 
              comparison[(2,2)]['count'],
              comparison[(3,3)]['count']]

bars = ax.bar(ngram_types, vocab_sizes, color=['steelblue', 'coral', 'gold'], alpha=0.7)
ax.set_ylabel('Vocabulary Size')
ax.set_title('N-gram Vocabulary Size Comparison')
ax.grid(axis='y', alpha=0.3)

for bar, size in zip(bars, vocab_sizes):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{size}', ha='center', va='bottom', fontweight='bold')

# 4. Top trigrams
ax = axes[1, 1]
trigrams, counts = zip(*top_trigrams[:6])
trigrams_short = [t[:25] + '...' if len(t) > 25 else t for t in trigrams]
ax.barh(trigrams_short, counts, color='lightcoral', alpha=0.7)
ax.set_xlabel('Frequency')
ax.set_title('Top 6 Trigrams')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('ngram_analysis.png', dpi=150, bbox_inches='tight')
print("\n📊 Gráfico salvo: ngram_analysis.png")

print("\n✅ N-gram analysis completo!")
print("\n💡 N-GRAM INSIGHTS:")
print("   - Unigrams: Capture individual word frequency")
print("   - Bigrams: Capture word pairs, better context")
print("   - Trigrams: Capture phrases, richer semantics")
print("   - Higher n = more context, but sparser data")
print("\n💡 WHEN TO USE:")
print("   - Unigrams: Simple classification, topic modeling")
print("   - Bigrams: Sentiment analysis, collocation extraction")
print("   - Trigrams: Named entity recognition, phrase matching")
print("   - Mixed (1,2) or (1,3): Balance between all")
print("\n💡 TRADE-OFFS:")
print("   - Higher n = exponentially more features")
print("   - Sparsity increases with n")
print("   - Computational cost grows")
print("   - Use max_features to limit vocabulary")
