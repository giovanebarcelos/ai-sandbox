#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GO1701-10VectorStores
Aula 17 - LLMs Locais com Ollama e RAG
Curso: Inteligência Artificial - FAPA

DEMONSTRAÇÃO COMPLETA: Vector Stores e Busca Semântica
=======================================================
Conceitos demonstrados:
  1. O que são embeddings (representações vetoriais de texto)
  2. Como funciona a busca por similaridade (cosine similarity)
  3. Vector Store: banco de dados otimizado para vetores
  4. Visualização do espaço vetorial com PCA/t-SNE
  5. ChromaDB como exemplo de vector store real

Não requer Ollama instalado - usa sklearn para embeddings TF-IDF.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# 1. O QUE SÃO EMBEDDINGS?
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("VECTOR STORES - DEMONSTRAÇÃO COMPLETA")
print("=" * 70)

# Base de conhecimento fictícia sobre Inteligência Artificial
# Cada documento representa um "chunk" de texto que seria armazenado no RAG
documentos = [
    # Tema: Machine Learning
    "Machine Learning é uma subárea da IA onde modelos aprendem padrões em dados.",
    "Redes neurais são modelos inspirados no cérebro humano usados em deep learning.",
    "Algoritmos de classificação como SVM e Random Forest são usados para categorizar dados.",
    "O treinamento supervisionado usa dados rotulados para ajustar os pesos do modelo.",
    "Overfitting ocorre quando o modelo memoriza os dados de treino mas não generaliza.",

    # Tema: RAG e LLMs
    "RAG combina busca de documentos com geração de texto por modelos de linguagem.",
    "Embeddings são representações vetoriais densas que capturam semântica do texto.",
    "ChromaDB é um banco de dados vetorial open-source ideal para aplicações RAG.",
    "LangChain é um framework que facilita a construção de pipelines com LLMs.",
    "FAISS do Meta permite busca vetorial eficiente em bilhões de documentos.",

    # Tema: Python e Programação
    "Python é a linguagem mais popular para ciência de dados e IA.",
    "NumPy oferece operações vetoriais eficientes para arrays multidimensionais.",
    "Pandas é usado para manipulação e análise de dados tabulares.",
    "TensorFlow e PyTorch são as principais bibliotecas para deep learning.",
    "Scikit-learn fornece implementações eficientes de algoritmos de ML clássico.",
]

# Rótulos temáticos para visualização
temas = (
    ["Machine Learning"] * 5 +
    ["RAG e LLMs"] * 5 +
    ["Python e Programação"] * 5
)

print("\n📚 Base de Conhecimento:")
for i, (doc, tema) in enumerate(zip(documentos, temas)):
    print(f"  [{i:2d}] [{tema:22s}] {doc[:65]}...")

# ═══════════════════════════════════════════════════════════════════
# 2. CRIAR EMBEDDINGS COM TF-IDF
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PASSO 1: Convertendo Documentos em Embeddings (Vetores)")
print("=" * 70)

# TF-IDF converte cada documento em um vetor numérico.
# Em produção, usaríamos modelos como nomic-embed-text ou all-MiniLM-L6-v2
# que capturam semântica melhor, mas TF-IDF serve para demonstração.
vectorizer = TfidfVectorizer(
    max_features=100,       # Vocabulário limitado a 100 termos mais frequentes
    ngram_range=(1, 2),     # Considerar uni-gramas e bi-gramas
    sublinear_tf=True       # Normalização logarítmica da frequência
)

# Matriz de embeddings: shape = (n_documentos, n_features)
# Cada linha é um vetor que representa um documento
embeddings_matriz = vectorizer.fit_transform(documentos).toarray()

print(f"\n✅ Embeddings criados:")
print(f"   Documentos: {embeddings_matriz.shape[0]}")
print(f"   Dimensões por vetor: {embeddings_matriz.shape[1]}")
print(f"\n📊 Exemplo - Vetor do documento 0 (primeiros 10 valores):")
print(f"   {embeddings_matriz[0, :10].round(4)}")
print(f"   (vetor completo tem {embeddings_matriz.shape[1]} dimensões)")

# ═══════════════════════════════════════════════════════════════════
# 3. VECTOR STORE: ARMAZENAMENTO E BUSCA
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PASSO 2: Vector Store - Armazenando e Buscando Vetores")
print("=" * 70)


class SimpleVectorStore:
    """
    Vector Store simplificado para fins didáticos.
    Em produção, usar ChromaDB, FAISS ou Pinecone.

    Estrutura interna:
        - documents[]: textos originais
        - embeddings[]: vetores correspondentes
        - metadatas[]: metadados opcionais (fonte, data, etc.)
    """

    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        self.documents = []
        self.embeddings = []
        self.metadatas = []

    def add(self, texts, metadatas=None):
        """Adiciona documentos e calcula seus embeddings."""
        vecs = self.vectorizer.transform(texts).toarray()
        self.documents.extend(texts)
        self.embeddings.extend(vecs)
        if metadatas:
            self.metadatas.extend(metadatas)
        else:
            self.metadatas.extend([{}] * len(texts))

    def similarity_search(self, query, k=3):
        """
        Busca os k documentos mais similares à query.

        Algoritmo:
          1. Converte query em vetor (embedding)
          2. Calcula cosine similarity entre query e todos os docs
          3. Retorna os top-k mais similares
        """
        # Converter query em vetor usando o mesmo vectorizer
        query_vec = self.vectorizer.transform([query]).toarray()

        # Cosine similarity: valores entre -1 e 1
        # 1.0 = idênticos, 0.0 = sem relação, -1.0 = opostos
        similarities = cosine_similarity(query_vec, self.embeddings)[0]

        # Ordenar por similaridade decrescente e pegar top-k
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'similarity': float(similarities[idx]),
                'metadata': self.metadatas[idx],
                'index': int(idx)
            })
        return results


# Criar e popular o vector store
store = SimpleVectorStore(vectorizer)
store.add(
    texts=documentos,
    metadatas=[{'tema': t, 'id': i} for i, t in enumerate(temas)]
)

print(f"\n✅ Vector Store criado com {len(store.documents)} documentos")

# Testar busca com diferentes queries
queries_teste = [
    "Como funciona o aprendizado de máquina?",
    "O que é RAG e para que serve?",
    "Quais bibliotecas Python usar para IA?",
]

print("\n" + "-" * 70)
print("TESTANDO BUSCA SEMÂNTICA:")
print("-" * 70)

resultados_por_query = {}
for query in queries_teste:
    print(f"\n🔍 Query: '{query}'")
    resultados = store.similarity_search(query, k=3)
    resultados_por_query[query] = resultados
    for rank, r in enumerate(resultados, 1):
        print(f"   {rank}. [{r['similarity']:.4f}] [{r['metadata']['tema']:22s}] "
              f"{r['document'][:60]}...")

# ═══════════════════════════════════════════════════════════════════
# 4. MATRIZ DE SIMILARIDADE
# ═══════════════════════════════════════════════════════════════════

# Calcular similaridade entre todos os documentos
# Isso revela a "estrutura semântica" do corpus
sim_matrix = cosine_similarity(embeddings_matriz)

# ═══════════════════════════════════════════════════════════════════
# 5. VISUALIZAÇÕES
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("VISUALIZAÇÕES DO ESPAÇO VETORIAL")
print("=" * 70)

# PCA para reduzir dimensionalidade: de 100d → 2d para visualizar
# PCA preserva a maior variância possível dos dados
pca = PCA(n_components=2, random_state=42)
coords_2d = pca.fit_transform(embeddings_matriz)
variancia_explicada = pca.explained_variance_ratio_.sum() * 100

print(f"\n📉 PCA: {embeddings_matriz.shape[1]}D → 2D")
print(f"   Variância explicada: {variancia_explicada:.1f}%")

# Cores por tema
cores_tema = {
    'Machine Learning': '#2196F3',
    'RAG e LLMs': '#4CAF50',
    'Python e Programação': '#FF9800',
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Vector Stores - Visualização do Espaço Vetorial', fontsize=14, fontweight='bold')

# --- Gráfico 1: Espaço vetorial 2D ---
ax1 = axes[0]
for tema_nome, cor in cores_tema.items():
    mascara = [t == tema_nome for t in temas]
    indices = np.where(mascara)[0]
    ax1.scatter(
        coords_2d[indices, 0],
        coords_2d[indices, 1],
        c=cor, label=tema_nome, s=150, alpha=0.8, zorder=3
    )
    # Numerar cada ponto
    for idx in indices:
        ax1.annotate(str(idx), (coords_2d[idx, 0], coords_2d[idx, 1]),
                     ha='center', va='center', fontsize=8, color='white', fontweight='bold')

# Marcar centróides dos temas
for tema_nome, cor in cores_tema.items():
    indices = [i for i, t in enumerate(temas) if t == tema_nome]
    cx = np.mean(coords_2d[indices, 0])
    cy = np.mean(coords_2d[indices, 1])
    ax1.scatter(cx, cy, c=cor, s=400, marker='*', edgecolors='black', linewidths=1.5, zorder=4)

ax1.set_title(f'Espaço Vetorial 2D (PCA)\n{variancia_explicada:.0f}% variância explicada')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.legend(loc='upper right', fontsize=8)
ax1.grid(True, alpha=0.3)

# Anotação explicativa
ax1.annotate(
    'Documentos do mesmo tema\nformam clusters naturais\n(★ = centróide do tema)',
    xy=(0.03, 0.03), xycoords='axes fraction',
    fontsize=8, style='italic',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8)
)

# --- Gráfico 2: Matriz de Similaridade ---
ax2 = axes[1]
im = ax2.imshow(sim_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax2, label='Cosine Similarity')

# Linhas divisórias entre temas
for sep in [5, 10]:
    ax2.axhline(y=sep - 0.5, color='black', linewidth=2)
    ax2.axvline(x=sep - 0.5, color='black', linewidth=2)

# Labels dos temas
ax2.set_xticks([2, 7, 12])
ax2.set_xticklabels(['ML', 'RAG', 'Python'], fontsize=9)
ax2.set_yticks([2, 7, 12])
ax2.set_yticklabels(['ML', 'RAG', 'Python'], fontsize=9)
ax2.set_title('Matriz de Similaridade\n(cosine similarity entre todos os docs)')

# Anotar alguns valores na diagonal de blocos
for row_block in range(3):
    for col_block in range(3):
        ri, ci = row_block * 5 + 2, col_block * 5 + 2
        val = sim_matrix[ri, ci]
        ax2.text(ci, ri, f'{val:.2f}', ha='center', va='center',
                 fontsize=9, color='black', fontweight='bold')

# --- Gráfico 3: Resultados de Busca ---
ax3 = axes[2]

# Mostrar scores das 3 queries de teste como barras agrupadas
query_labels = ['Query ML', 'Query RAG', 'Query Python']
cores_rank = ['#1f77b4', '#ff7f0e', '#2ca02c']
n_queries = len(queries_teste)
n_resultados = 3

x_base = np.arange(n_queries)
largura = 0.25

for rank in range(n_resultados):
    scores = [resultados_por_query[q][rank]['similarity'] for q in queries_teste]
    ax3.bar(x_base + rank * largura, scores, largura,
            label=f'Top-{rank+1}', color=cores_rank[rank], alpha=0.8)

ax3.set_xticks(x_base + largura)
ax3.set_xticklabels(query_labels, fontsize=9)
ax3.set_ylabel('Cosine Similarity')
ax3.set_ylim(0, 1.1)
ax3.set_title('Scores de Busca por Query\n(Top-3 resultados por query)')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')
ax3.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Threshold típico (0.3)')

for bars in ax3.containers:
    ax3.bar_label(bars, fmt='%.2f', fontsize=7)

plt.tight_layout()
plt.savefig('vector_stores_demo.png', dpi=120, bbox_inches='tight')
plt.show()
print("✅ Gráfico salvo: vector_stores_demo.png")

# ═══════════════════════════════════════════════════════════════════
# 6. DEMO CHROMADB (OPCIONAL - instalar com: pip install chromadb)
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("DEMO CHROMADB (Vector Store Real)")
print("=" * 70)

try:
    import chromadb

    # ChromaDB em memória (sem persistência, ideal para testes)
    chroma_client = chromadb.Client()

    # Criar coleção - equivale a uma "tabela" no banco relacional
    collection = chroma_client.get_or_create_collection(
        name="aula17_demo",
        metadata={"description": "Demo de Vector Store - Aula 17"}
    )

    # Adicionar documentos com IDs únicos e metadados
    collection.add(
        ids=[f"doc_{i}" for i in range(len(documentos))],
        documents=documentos,
        metadatas=[{"tema": t, "indice": i} for i, t in enumerate(temas)]
    )

    print(f"\n✅ ChromaDB: {collection.count()} documentos indexados")

    # Busca semântica com ChromaDB
    query_chroma = "aprendizado de máquina e redes neurais"
    results = collection.query(
        query_texts=[query_chroma],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    print(f"\n🔍 Query ChromaDB: '{query_chroma}'")
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        dist = results['distances'][0][i]
        # ChromaDB usa distância L2 (menor = mais similar)
        print(f"   {i+1}. [dist={dist:.4f}] [{meta['tema']:22s}] {doc[:55]}...")

    print("\n💡 ChromaDB vs Implementação Manual:")
    print("   • ChromaDB usa embeddings reais (sentence-transformers internamente)")
    print("   • Suporta persistência em disco (persist_directory=...)")
    print("   • Escala para milhões de documentos com HNSW")
    print("   • Compatible com LangChain e LlamaIndex")

except ImportError:
    print("\n⚠️  ChromaDB não instalado.")
    print("   Para instalar: pip install chromadb")
    print("   Nossa implementação manual acima funciona sem dependências extras!")

# ═══════════════════════════════════════════════════════════════════
# 7. RESUMO CONCEITUAL
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("RESUMO - VECTOR STORES")
print("=" * 70)
print("""
📌 PIPELINE COMPLETO:

  Texto → [Embedding Model] → Vetor Numérico → [Vector Store] → Busca por Similaridade

📌 MÉTRICAS DE SIMILARIDADE:
  • Cosine Similarity: ângulo entre vetores (0=diferente, 1=idêntico)
  • L2 Distance: distância euclidiana (menor = mais similar)
  • Dot Product: produto interno (mais rápido, requer normalização)

📌 VECTOR STORES POPULARES:
  • ChromaDB: open-source, local, ideal para desenvolvimento
  • FAISS: Meta, em memória, ultra-rápido para produção
  • Pinecone: cloud, escalável (pay-per-use)
  • Weaviate: open-source com GraphQL
  • Qdrant: Rust, alta performance, open-source

📌 CHUNKING (Divisão de Documentos):
  • Tamanho típico: 200-500 tokens por chunk
  • Overlap: 20-50 tokens (manter contexto entre chunks)
  • Estratégia: RecursiveCharacterTextSplitter (LangChain)
""")
