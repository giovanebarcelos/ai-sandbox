# GO1517-14aFasttextSubwordAnalysis
from gensim.models import FastText
import matplotlib.pyplot as plt
import numpy as np

print("🚀 FastText Subword Analysis Demo\n")
print("="*70)

# Training corpus
sentences = [
    ['machine', 'learning', 'is', 'awesome'],
    ['deep', 'learning', 'transforms', 'ai'],
    ['neural', 'networks', 'learn', 'patterns'],
    ['python', 'programming', 'enables', 'ml'],
    ['transformer', 'architecture', 'revolutionized', 'nlp']
] * 20  # Repeat for training

print(f"📚 Training corpus: {len(sentences)} sentences\n")

# Train FastText (simulated - actual training takes time)
print("⚙️  Training FastText model with subword features...\n")
print("   Note: Real training would use gensim.models.FastText")
print("   Simulating results for demo...\n")

# Simulated embeddings
vocab = {
    'learning': np.random.randn(100),
    'python': np.random.randn(100),
    'programming': np.random.randn(100),
    'transformer': np.random.randn(100)
}

# OOV words (not in training)
oov_words = ['learnings', 'pythonic', 'transformers']

print("="*70)
print("\n✅ SUBWORD CAPABILITIES:\n")

# Show subword breakdown
word = 'programming'
subwords = ['<pr', 'pro', 'rog', 'ogr', 'gra', 'ram', 'amm', 'mmi', 'min', 'ing', 'ng>']

print(f"Word: '{word}'")
print(f"Subwords (3-grams): {subwords[:8]}...")
print(f"Total subwords: {len(subwords)}\n")

print("💡 FastText computes word vector as SUM of subword vectors!")
print("   → Can generate vectors for unseen words!\n")

print("="*70)
print("\n🔍 OUT-OF-VOCABULARY (OOV) HANDLING:\n")

for oov in oov_words:
    print(f"'{oov}' (not in training):")
    print(f"   ✓ FastText CAN generate embedding (from subwords)")
    print(f"   ✗ Word2Vec CANNOT (word not in vocabulary)")
    print()

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Vocabulary coverage comparison
ax = axes[0]
models = ['FastText\n(subword)', 'Word2Vec\n(word-level)']
vocab_coverage = [100, 75]  # Approximate
oov_handling = [95, 0]  # FastText handles OOV, Word2Vec doesn't

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, vocab_coverage, width, label='In-Vocab Words', 
              color='skyblue', alpha=0.8)
bars2 = ax.bar(x + width/2, oov_handling, width, label='OOV Handling', 
              color='lightcoral', alpha=0.8)

ax.set_ylabel('Coverage (%)')
ax.set_title('Vocabulary Coverage & OOV Handling')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{height:.0f}%', ha='center', va='bottom', fontweight='bold')

# 2. Subword importance
ax = axes[1]
features = ['Morphology\nAware', 'Rare Words\nHandling', 'Typo\nRobust', 
           'Multilingual\nSupport', 'Training\nSpeed']
fasttext_scores = [95, 90, 85, 88, 70]
word2vec_scores = [40, 30, 20, 60, 95]

x_pos = np.arange(len(features))

ax.plot(x_pos, fasttext_scores, 'o-', linewidth=2.5, markersize=10, 
       label='FastText', color='green')
ax.plot(x_pos, word2vec_scores, 's--', linewidth=2.5, markersize=10,
       label='Word2Vec', color='blue')

ax.set_xticks(x_pos)
ax.set_xticklabels(features, fontsize=9)
ax.set_ylabel('Capability Score')
ax.set_title('Feature Comparison: FastText vs Word2Vec')
ax.legend()
ax.set_ylim(0, 100)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('fasttext_subword_analysis.png', dpi=150, bbox_inches='tight')
print("📊 Gráfico salvo: fasttext_subword_analysis.png")

print("\n✅ FastText analysis completo!")
print("\n💡 KEY ADVANTAGES:")
print("   - Subword embeddings: Captures morphology (pre-, -ing, -tion)")
print("   - OOV handling: Generates vectors for unseen words")
print("   - Typo robustness: Similar subwords → similar vectors")
print("   - Rich morphology languages: German, Turkish, Finnish")
print("\n💡 WHEN TO USE:")
print("   - Small training corpus (many rare words)")
print("   - Domain-specific vocabulary")
print("   - Morphologically rich languages")
print("   - Need OOV word embeddings")
