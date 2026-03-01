# GO1717-30aHybridSearchBm25VectorSearch
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class HybridSearchRAG:
    """
    Hybrid Search combina:
    1. BM25 (busca léxica) - bom para keywords exatas
    2. Vector search (semântica) - bom para significado

    Fusion strategies:
    - Reciprocal Rank Fusion (RRF)
    - Weighted Score Combination
    - Rank-based reordering
    """

    def __init__(self, documents: List[Dict], alpha: float = 0.5):
        """
        Args:
            documents: Lista de docs com 'text' e 'metadata'
            alpha: Peso BM25 vs vector (0=só vector, 1=só BM25)
        """
        self.documents = documents
        self.alpha = alpha

        # BM25 setup
        tokenized_docs = [doc['text'].lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

        # Vector search setup (mock embeddings)
        print("Generating embeddings...")
        self.embeddings = self._generate_embeddings()

        print(f"✅ Hybrid search initialized:")
        print(f"   {len(documents)} documents")
        print(f"   Alpha (BM25 weight): {alpha}")

    def _generate_embeddings(self):
        """Generate mock embeddings (384-dim)"""
        embeddings = []
        for doc in self.documents:
            # Simple mock: use word counts as features
            words = doc['text'].lower().split()
            # Create pseudo-embedding
            emb = [hash(w) % 100 / 100.0 for w in words[:384]]
            # Pad to 384 dims
            emb = (emb + [0] * 384)[:384]
            embeddings.append(emb)
        return np.array(embeddings)

    def _bm25_search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """BM25 léxico search"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:k]
        results = [(int(idx), float(scores[idx])) for idx in top_indices]

        return results

    def _vector_search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Semantic vector search"""
        # Generate query embedding (mock)
        words = query.lower().split()
        query_emb = [hash(w) % 100 / 100.0 for w in words[:384]]
        query_emb = np.array((query_emb + [0] * 384)[:384])

        # Cosine similarity
        similarities = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-10
        )

        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]

        return results

    def _reciprocal_rank_fusion(self, 
                                 results1: List[Tuple[int, float]], 
                                 results2: List[Tuple[int, float]],
                                 k: int = 60) -> List[Tuple[int, float]]:
        """
        Reciprocal Rank Fusion (RRF)

        RRF score = 1/(k + rank)
        Combina rankings de múltiplas fontes
        """
        rrf_scores = {}

        # Process first ranking
        for rank, (doc_id, score) in enumerate(results1, 1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

        # Process second ranking
        for rank, (doc_id, score) in enumerate(results2, 1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        return [(doc_id, score) for doc_id, score in sorted_results]

    def search(self, query: str, k: int = 5, method: str = 'hybrid') -> List[Dict]:
        """
        Hybrid search com múltiplas estratégias

        Args:
            query: Search query
            k: Number of results
            method: 'bm25', 'vector', 'hybrid', 'rrf'
        """
        if method == 'bm25':
            results = self._bm25_search(query, k)

        elif method == 'vector':
            results = self._vector_search(query, k)

        elif method == 'rrf':
            # Reciprocal Rank Fusion
            bm25_results = self._bm25_search(query, k * 2)
            vector_results = self._vector_search(query, k * 2)
            results = self._reciprocal_rank_fusion(bm25_results, vector_results)[:k]

        else:  # 'hybrid' - weighted combination
            bm25_results = dict(self._bm25_search(query, k * 2))
            vector_results = dict(self._vector_search(query, k * 2))

            # Combine scores
            all_doc_ids = set(bm25_results.keys()) | set(vector_results.keys())

            combined_scores = {}
            for doc_id in all_doc_ids:
                bm25_score = bm25_results.get(doc_id, 0)
                vector_score = vector_results.get(doc_id, 0)

                # Normalize and combine
                combined = self.alpha * bm25_score + (1 - self.alpha) * vector_score
                combined_scores[doc_id] = combined

            # Sort and take top k
            sorted_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            results = sorted_ids[:k]

        # Return documents with scores
        return [
            {
                **self.documents[doc_id],
                'score': score,
                'rank': rank + 1
            }
            for rank, (doc_id, score) in enumerate(results)
        ]

# === DEMO ===

# Create documents
documents = [
    {'text': 'Machine learning is a subset of artificial intelligence', 'metadata': {'source': 'ml_basics.pdf'}},
    {'text': 'Neural networks are computational models inspired by biological neurons', 'metadata': {'source': 'nn_intro.pdf'}},
    {'text': 'Deep learning uses multiple layers to learn representations', 'metadata': {'source': 'dl_guide.pdf'}},
    {'text': 'Reinforcement learning trains agents through rewards', 'metadata': {'source': 'rl_tutorial.pdf'}},
    {'text': 'Natural language processing enables computers to understand human language', 'metadata': {'source': 'nlp_basics.pdf'}},
    {'text': 'Computer vision allows machines to interpret visual information', 'metadata': {'source': 'cv_intro.pdf'}},
    {'text': 'Transformers revolutionized NLP with attention mechanisms', 'metadata': {'source': 'transformers.pdf'}},
    {'text': 'Convolutional neural networks excel at image recognition tasks', 'metadata': {'source': 'cnn_guide.pdf'}},
]

print("🔍 Testando Hybrid Search\n")

# Test different alpha values
queries = [
    "neural networks deep learning",  # Both keywords and semantic
    "how do agents learn",  # More semantic
    "CNN image",  # Exact keywords
]

results_data = []

for query in queries:
    print(f"\n📌 Query: '{query}'")
    print("="*70)

    for method in ['bm25', 'vector', 'rrf', 'hybrid']:
        searcher = HybridSearchRAG(documents, alpha=0.5)
        results = searcher.search(query, k=3, method=method)

        print(f"\n{method.upper()}:")
        for i, doc in enumerate(results, 1):
            text_preview = doc['text'][:50] + '...'
            print(f"  {i}. [score={doc['score']:.3f}] {text_preview}")

            results_data.append({
                'query': query,
                'method': method,
                'rank': i,
                'score': doc['score'],
                'doc': text_preview
            })

# Visualize comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Score distribution by method
ax = axes[0, 0]
import pandas as pd
df = pd.DataFrame(results_data)
for method in ['bm25', 'vector', 'rrf', 'hybrid']:
    method_df = df[df['method'] == method]
    ax.plot(method_df.groupby('query')['score'].mean(), 
            marker='o', label=method, linewidth=2)
ax.set_xlabel('Query')
ax.set_ylabel('Average Score')
ax.set_title('Score Distribution by Method')
ax.legend()
ax.grid(alpha=0.3)

# 2. Alpha sensitivity (hybrid only)
ax = axes[0, 1]
alphas = np.linspace(0, 1, 11)
avg_scores = []

for alpha in alphas:
    searcher = HybridSearchRAG(documents, alpha=alpha)
    scores = []
    for q in queries:
        results = searcher.search(q, k=3, method='hybrid')
        scores.extend([r['score'] for r in results])
    avg_scores.append(np.mean(scores))

ax.plot(alphas, avg_scores, marker='o', linewidth=2, color='purple')
ax.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='α=0.5 (balanced)')
ax.set_xlabel('Alpha (BM25 weight)')
ax.set_ylabel('Average Score')
ax.set_title('Hybrid Search: Alpha Sensitivity')
ax.legend()
ax.grid(alpha=0.3)
ax.annotate('Pure vector', xy=(0, avg_scores[0]), xytext=(0.1, avg_scores[0]-0.02),
            arrowprops=dict(arrowstyle='->', color='blue'))
ax.annotate('Pure BM25', xy=(1, avg_scores[-1]), xytext=(0.85, avg_scores[-1]+0.02),
            arrowprops=dict(arrowstyle='->', color='red'))

# 3. Method comparison heatmap
ax = axes[1, 0]
pivot = df.pivot_table(values='score', index='method', columns='query', aggfunc='mean')
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Score'})
ax.set_title('Average Scores: Method vs Query')
ax.set_xlabel('Query')
ax.set_ylabel('Method')

# 4. Rank consistency
ax = axes[1, 1]
methods = df['method'].unique()
rank_stds = []
for method in methods:
    method_df = df[df['method'] == method]
    rank_std = method_df.groupby('query')['rank'].std().mean()
    rank_stds.append(rank_std)

bars = ax.barh(methods, rank_stds, color='skyblue', alpha=0.7)
ax.set_xlabel('Rank Std Dev (lower = more consistent)')
ax.set_title('Ranking Consistency Across Queries')
ax.grid(axis='x', alpha=0.3)

# Annotate bars
for bar, std in zip(bars, rank_stds):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2, 
            f'{std:.3f}', ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('hybrid_search_comparison.png', dpi=150, bbox_inches='tight')
print("\n\n📊 Gráfico salvo: hybrid_search_comparison.png")

# Summary
print("\n📈 RECOMENDAÇÕES:")
print("- Use BM25 para queries com keywords exatas")
print("- Use vector search para queries conceituais")
print("- Use RRF quando precisar de diversidade")
print("- Use hybrid com α=0.5 para caso geral")
print("- Ajuste α baseado no seu domínio (teste A/B)")

print("\n✅ Hybrid Search implementado!")
