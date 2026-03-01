# GO1519-NãoRequerInstalaçãoDeErro
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

class WordEmbeddingAnalyzer:
    """
    Analyze word embeddings: similarity, analogies, visualization

    Capabilities:
    - Semantic similarity
    - Word analogies (king - man + woman = queen)
    - Clustering similar words
    - 2D visualization with t-SNE
    """

    def __init__(self, sentences=None, model_path=None):
        if model_path:
            self.model = Word2Vec.load(model_path)
        elif sentences:
            # Train simple model
            self.model = Word2Vec(
                sentences=sentences,
                vector_size=100,
                window=5,
                min_count=1,
                workers=4,
                epochs=10
            )
        else:
            raise ValueError("Provide sentences or model_path")

        self.vocab = list(self.model.wv.index_to_key)
        print(f"✅ Model loaded/trained with {len(self.vocab)} words")

    def find_similar(self, word: str, top_n: int = 10):
        """Find most similar words"""
        try:
            similar = self.model.wv.most_similar(word, topn=top_n)
            return similar
        except KeyError:
            return f"Word '{word}' not in vocabulary"

    def word_analogy(self, positive: list, negative: list, top_n: int = 5):
        """
        Solve word analogies
        Example: king - man + woman = queen
        positive=['king', 'woman'], negative=['man']
        """
        try:
            result = self.model.wv.most_similar(
                positive=positive,
                negative=negative,
                topn=top_n
            )
            return result
        except KeyError as e:
            return f"Word not in vocabulary: {e}"

    def compute_similarity(self, word1: str, word2: str):
        """Compute cosine similarity between two words"""
        try:
            sim = self.model.wv.similarity(word1, word2)
            return sim
        except KeyError:
            return None

    def visualize_embeddings(self, words: list = None, n_samples: int = 50):
        """Visualize word embeddings in 2D using t-SNE"""
        if words is None:
            # Sample random words
            words = np.random.choice(self.vocab, min(n_samples, len(self.vocab)), replace=False)

        # Get vectors
        vectors = [self.model.wv[word] for word in words]

        # Reduce to 2D
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(words)-1))
        vectors_2d = tsne.fit_transform(vectors)

        return words, vectors_2d

# === DEMO ===

print("🔬 Word Embedding Analysis\n")
print("="*70)

# Training corpus
sentences = [
    ['python', 'programming', 'language', 'coding'],
    ['java', 'programming', 'language', 'development'],
    ['machine', 'learning', 'artificial', 'intelligence'],
    ['deep', 'learning', 'neural', 'networks'],
    ['data', 'science', 'analysis', 'visualization'],
    ['computer', 'science', 'algorithms', 'data', 'structures'],
    ['web', 'development', 'frontend', 'backend'],
    ['database', 'sql', 'queries', 'data'],
    ['ai', 'artificial', 'intelligence', 'machine', 'learning'],
    ['neural', 'networks', 'deep', 'learning', 'ai'],
    ['king', 'queen', 'man', 'woman', 'royalty'],
    ['doctor', 'nurse', 'hospital', 'medical'],
    ['france', 'paris', 'germany', 'berlin'],
    ['cat', 'dog', 'animal', 'pet'],
]

analyzer = WordEmbeddingAnalyzer(sentences=sentences)

print("\n📊 1. WORD SIMILARITY\n")

test_words = ['python', 'machine', 'neural']
for word in test_words:
    print(f"Words similar to '{word}':")
    similar = analyzer.find_similar(word, top_n=5)
    if isinstance(similar, list):
        for w, score in similar:
            print(f"   {w}: {score:.3f}")
    print()

print("="*70)
print("\n🧮 2. WORD ANALOGIES\n")

analogies = [
    (['king', 'woman'], ['man'], 'king - man + woman = ?'),
    (['paris', 'germany'], ['france'], 'paris - france + germany = ?'),
    (['deep', 'networks'], ['neural'], 'deep - neural + networks = ?'),
]

for positive, negative, description in analogies:
    print(f"{description}")
    result = analyzer.word_analogy(positive, negative, top_n=3)
    if isinstance(result, list):
        for word, score in result:
            print(f"   {word}: {score:.3f}")
    else:
        print(f"   {result}")
    print()

print("="*70)
print("\n📏 3. PAIRWISE SIMILARITIES\n")

word_pairs = [
    ('python', 'java'),
    ('machine', 'learning'),
    ('cat', 'dog'),
    ('python', 'database'),
]

for w1, w2 in word_pairs:
    sim = analyzer.compute_similarity(w1, w2)
    if sim:
        print(f"Similarity({w1}, {w2}) = {sim:.3f}")

# Visualize
print("\n" + "="*70)
print("\n📊 4. VISUALIZATION\n")

words_to_plot = ['python', 'java', 'machine', 'learning', 'deep', 'neural',
                 'data', 'science', 'king', 'queen', 'man', 'woman',
                 'cat', 'dog', 'paris', 'berlin']

words, vectors_2d = analyzer.visualize_embeddings(words_to_plot)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1. t-SNE plot
ax = axes[0]
ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.6, s=100, c='steelblue')

for i, word in enumerate(words):
    ax.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]),
               xytext=(5, 2), textcoords='offset points',
               fontsize=9, fontweight='bold')

ax.set_title('Word Embeddings (t-SNE Projection)')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.grid(alpha=0.3)

# 2. Similarity heatmap
ax = axes[1]

selected_words = ['python', 'java', 'machine', 'learning', 'king', 'queen']
sim_matrix = np.zeros((len(selected_words), len(selected_words)))

for i, w1 in enumerate(selected_words):
    for j, w2 in enumerate(selected_words):
        sim = analyzer.compute_similarity(w1, w2)
        sim_matrix[i, j] = sim if sim else 0

im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=0, vmax=1)
ax.set_xticks(range(len(selected_words)))
ax.set_yticks(range(len(selected_words)))
ax.set_xticklabels(selected_words, rotation=45, ha='right')
ax.set_yticklabels(selected_words)
ax.set_title('Word Similarity Matrix')

for i in range(len(selected_words)):
    for j in range(len(selected_words)):
        text = ax.text(j, i, f'{sim_matrix[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=9)

plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('word_similarity_analysis.png', dpi=150, bbox_inches='tight')
print("📊 Gráfico salvo: word_similarity_analysis.png")

print("\n✅ Word embedding analysis completo!")
print("\n💡 KEY INSIGHTS:")
print("   - Word2Vec captures semantic relationships")
print("   - Similar words cluster together in vector space")
print("   - Analogies work through vector arithmetic")
print("   - Cosine similarity measures semantic closeness")
print("\n💡 APPLICATIONS:")
print("   - Document similarity")
print("   - Search and retrieval")
print("   - Recommendation systems")
print("   - Feature engineering for ML")
