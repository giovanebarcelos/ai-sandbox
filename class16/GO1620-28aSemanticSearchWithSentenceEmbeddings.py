# GO1620-28aSemanticSearchWithSentenceEmbeddings
from sentence_transformers import SentenceTransformer, util
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

class SemanticSearchEngine:
    """
    Busca semântica usando embeddings Sentence-BERT

    Funcionalidades:
    - Codificar documentos em vetores densos
    - Busca rápida por similaridade
    - Visualizar espaço de embedding
    - Comparar diferentes consultas
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"🔍 Loading Sentence Transformer: {model_name}...\n")
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        print(f"✅ Modelo carregado! Dimensão do embedding: {self.model.get_sentence_embedding_dimension()}\n")

    def index_documents(self, documents: list):
        """Codificar e indexar documentos"""
        print(f"📚 Indexando {len(documents)} documentos...\n")

        self.documents = documents
        self.embeddings = self.model.encode(documents, convert_to_tensor=True)

        print(f"✅ Indexados {len(documents)} documentos")
        print(f"   Embedding shape: {self.embeddings.shape}\n")

    def search(self, query: str, top_k: int = 5):
        """Buscar documentos mais similares"""
        if self.embeddings is None:
            raise ValueError("No documents indexed! Call index_documents() first.")

        # Encode query
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Compute cosine similarity
        cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]

        # Get top-k results
        top_results = np.argsort(-cos_scores.cpu().numpy())[:top_k]

        results = []
        for idx in top_results:
            results.append({
                'document': self.documents[idx],
                'score': float(cos_scores[idx]),
                'rank': len(results) + 1
            })

        return results

    def compare_queries(self, queries: list, top_k: int = 3):
        """Comparar múltiplas consultas"""
        print("🔍 Comparando Consultas:\n")

        for query in queries:
            print(f"Consulta: \"{query}\"")
            results = self.search(query, top_k=top_k)

            for result in results:
                print(f"   {result['rank']}. [{result['score']:.3f}] {result['document'][:80]}...")
            print()

    def visualize_embeddings(self, queries: list = None):
        """Visualizar embeddings de documentos e consultas em 2D"""
        if self.embeddings is None:
print("⚠️  Nenhum documento indexado!")
            return

        # Convert to numpy
        doc_embeddings = self.embeddings.cpu().numpy()

        # Add queries if provided
        if queries:
            query_embeddings = self.model.encode(queries)
            all_embeddings = np.vstack([doc_embeddings, query_embeddings])
            labels = ['Doc'] * len(self.documents) + ['Query'] * len(queries)
        else:
            all_embeddings = doc_embeddings
            labels = ['Doc'] * len(self.documents)

        # Reduce to 2D using t-SNE
        print("📊 Reduzindo embeddings para 2D com t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)-1))
        embeddings_2d = tsne.fit_transform(all_embeddings)

        return embeddings_2d, labels

# === DEMO ===

print("🔍 Demo de Motor de Busca Semântica\n")
print("="*70)

# Sample documents
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Python is a popular programming language for data science.",
    "Natural language processing enables computers to understand human language.",
    "Computer vision allows machines to interpret visual information.",
    "Reinforcement learning trains agents through rewards and penalties.",
    "Transformers revolutionized NLP with attention mechanisms.",
    "Convolutional neural networks excel at image recognition tasks.",
    "Recurrent neural networks process sequential data effectively.",
    "Supervised learning requires labeled training data.",
    "Unsupervised learning discovers patterns in unlabeled data.",
    "Transfer learning leverages pre-trained models for new tasks.",
    "GPT models generate human-like text using transformers.",
    "BERT bidirectional encoder improves language understanding.",
    "Object detection identifies and locates objects in images.",
]

# Create search engine
search_engine = SemanticSearchEngine()
search_engine.index_documents(documents)

# Test queries
test_queries = [
    "How do neural networks learn?",
    "What is NLP?",
    "Image processing techniques"
]

print("\n" + "="*70)
search_engine.compare_queries(test_queries, top_k=3)

# Similarity matrix
print("="*70)
print("\n📊 Calculando Matriz de Similaridade...\n")

# Encode all documents
doc_embeddings = search_engine.embeddings

# Compute pairwise similarities
similarity_matrix = util.cos_sim(doc_embeddings, doc_embeddings).cpu().numpy()

print(f"Forma da matriz de similaridade: {similarity_matrix.shape}")
print(f"Similaridade média: {similarity_matrix.mean():.3f}")
print(f"Similaridade máxima (excluindo diagonal): {np.max(similarity_matrix - np.eye(len(similarity_matrix))):.3f}\n")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# 1. Similarity heatmap
ax = axes[0, 0]
sns.heatmap(similarity_matrix, cmap='RdYlGn', vmin=0, vmax=1, 
            square=True, ax=ax, cbar_kws={'label': 'Similaridade de Cosseno'})
ax.set_title('Matriz de Similaridade de Documentos')
ax.set_xlabel('Índice do Documento')
ax.set_ylabel('Índice do Documento')

# 2. Query comparison
ax = axes[0, 1]
query_results = {}
for query in test_queries:
    results = search_engine.search(query, top_k=5)
    scores = [r['score'] for r in results]
    query_results[query[:30] + "..."] = scores

x = np.arange(5)
width = 0.25

for i, (query, scores) in enumerate(query_results.items()):
    ax.bar(x + i*width, scores, width, label=query, alpha=0.8)

ax.set_xlabel('Rank')
ax.set_ylabel('Pontuação de Similaridade')
ax.set_title('Top-5 Resultados por Consulta')
ax.set_xticks(x + width)
ax.set_xticklabels([f'Top {i+1}' for i in range(5)])
ax.legend(fontsize=7)
ax.grid(axis='y', alpha=0.3)

# 3. t-SNE visualization
ax = axes[1, 0]
embeddings_2d, labels = search_engine.visualize_embeddings(test_queries)

doc_points = embeddings_2d[:len(documents)]
query_points = embeddings_2d[len(documents):]

ax.scatter(doc_points[:, 0], doc_points[:, 1], 
          c='blue', alpha=0.6, s=100, label='Documentos', marker='o')
ax.scatter(query_points[:, 0], query_points[:, 1], 
          c='red', alpha=0.8, s=200, label='Consultas', marker='*')

# Annotate queries
for i, query in enumerate(test_queries):
    ax.annotate(f"Q{i+1}", (query_points[i, 0], query_points[i, 1]),
               xytext=(5, 5), textcoords='offset points', 
               fontweight='bold', fontsize=10, color='red')

ax.set_title('Espaço de Embedding (t-SNE)')
ax.set_xlabel('Dimensão 1')
ax.set_ylabel('Dimensão 2')
ax.legend()
ax.grid(alpha=0.3)

# 4. Similarity distribution
ax = axes[1, 1]

# Get upper triangle (exclude diagonal)
mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
similarities_upper = similarity_matrix[mask]

ax.hist(similarities_upper, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
ax.axvline(similarities_upper.mean(), color='red', linestyle='--', 
          linewidth=2, label=f'Média: {similarities_upper.mean():.3f}')
ax.set_xlabel('Similaridade de Cosseno')
ax.set_ylabel('Frequência')
ax.set_title('Distribuição de Similaridades entre Documentos')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
print("📊 Gráfico salvo: semantic_search.png")

print("\n✅ Semantic search demo completo!")
print("\n💡 CONCEITOS-CHAVE:")
print("   - Sentence-BERT: BERT ajustado para similaridade semântica")
print("   - Similaridade de cosseno: Mede similaridade vetorial (-1 a 1)")
print("   - Recuperação densa: Superior à busca por palavras-chave")
print("   - Aplicações: Correspondência de FAQ, detecção de duplicatas, recomendação")
print("\n💡 DICAS DE DESEMPENHO:")
print("   - Use vizinho mais próximo aproximado (FAISS, Annoy) em grande escala")
print("   - Armazene embeddings em cache para evitar recodificação")
print("   - Codificação em lote para maior eficiência")
print("   - Considere fine-tuning específico de domínio")
