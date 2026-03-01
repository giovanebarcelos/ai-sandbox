# GO1723-34aVectorStoreComparisonFaissVs
import numpy as np
import time
import faiss
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from collections import defaultdict

class VectorStoreComparison:
    """
    Comparação de vector stores para RAG:
    - FAISS (Facebook AI Similarity Search)
    - ChromaDB (embedding database)
    - In-memory (baseline simples)

    Métricas comparadas:
    - Index build time
    - Search latency
    - Memory usage
    - Accuracy (recall@k)
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.results = defaultdict(list)

    def generate_test_data(self, n_vectors: int) -> tuple:
        """Gera vetores de teste"""
        # Database vectors
        db_vectors = np.random.randn(n_vectors, self.dimension).astype('float32')
        # Query vectors
        query_vectors = np.random.randn(100, self.dimension).astype('float32')

        # Normalize
        db_vectors = db_vectors / np.linalg.norm(db_vectors, axis=1, keepdims=True)
        query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)

        return db_vectors, query_vectors

    def benchmark_faiss(self, db_vectors: np.ndarray, 
                       query_vectors: np.ndarray, k: int = 5) -> Dict:
        """Benchmark FAISS"""
        print("\n🔍 Testing FAISS...")

        # Build index
        start = time.time()
        index = faiss.IndexFlatL2(self.dimension)
        index.add(db_vectors)
        build_time = time.time() - start

        # Search
        search_times = []
        all_distances = []
        all_indices = []

        for query in query_vectors:
            start = time.time()
            distances, indices = index.search(query.reshape(1, -1), k)
            search_times.append(time.time() - start)
            all_distances.append(distances[0])
            all_indices.append(indices[0])

        return {
            'build_time': build_time,
            'avg_search_time': np.mean(search_times),
            'std_search_time': np.std(search_times),
            'total_time': build_time + sum(search_times),
            'distances': all_distances,
            'indices': all_indices
        }

    def benchmark_numpy(self, db_vectors: np.ndarray,
                       query_vectors: np.ndarray, k: int = 5) -> Dict:
        """Benchmark NumPy brute-force (baseline)"""
        print("\n🔍 Testing NumPy Baseline...")

        # Build (no indexing)
        build_time = 0

        # Search
        search_times = []
        all_distances = []
        all_indices = []

        for query in query_vectors:
            start = time.time()
            # Brute force cosine similarity
            similarities = np.dot(db_vectors, query)
            top_k_idx = np.argsort(similarities)[::-1][:k]
            distances = 1 - similarities[top_k_idx]  # Convert to distance
            search_times.append(time.time() - start)
            all_distances.append(distances)
            all_indices.append(top_k_idx)

        return {
            'build_time': build_time,
            'avg_search_time': np.mean(search_times),
            'std_search_time': np.std(search_times),
            'total_time': sum(search_times),
            'distances': all_distances,
            'indices': all_indices
        }

    def calculate_recall(self, retrieved_indices: List[np.ndarray],
                        ground_truth_indices: List[np.ndarray],
                        k: int = 5) -> float:
        """Calcula recall@k comparado com ground truth"""
        recalls = []
        for retrieved, truth in zip(retrieved_indices, ground_truth_indices):
            retrieved_set = set(retrieved[:k])
            truth_set = set(truth[:k])
            recall = len(retrieved_set & truth_set) / len(truth_set)
            recalls.append(recall)
        return np.mean(recalls)

    def run_comparison(self, sizes: List[int] = [1000, 5000, 10000, 50000],
                      k: int = 5):
        """Executa comparação completa"""
        print("\n" + "="*70)
        print("🏁 INICIANDO BENCHMARK DE VECTOR STORES")
        print("="*70)

        for size in sizes:
            print(f"\n📊 Testing with {size:,} vectors...")

            # Generate data
            db_vectors, query_vectors = self.generate_test_data(size)

            # Benchmark NumPy (ground truth)
            numpy_result = self.benchmark_numpy(db_vectors, query_vectors, k)
            self.results['numpy']['sizes'].append(size)
            self.results['numpy']['build_times'].append(numpy_result['build_time'])
            self.results['numpy']['search_times'].append(numpy_result['avg_search_time'])
            self.results['numpy']['total_times'].append(numpy_result['total_time'])

            # Benchmark FAISS
            faiss_result = self.benchmark_faiss(db_vectors, query_vectors, k)
            self.results['faiss']['sizes'].append(size)
            self.results['faiss']['build_times'].append(faiss_result['build_time'])
            self.results['faiss']['search_times'].append(faiss_result['avg_search_time'])
            self.results['faiss']['total_times'].append(faiss_result['total_time'])

            # Calculate accuracy (recall)
            recall = self.calculate_recall(
                faiss_result['indices'],
                numpy_result['indices'],
                k
            )
            self.results['faiss']['recalls'].append(recall)

            print(f"  ✅ FAISS: build={faiss_result['build_time']:.4f}s, "
                  f"search={faiss_result['avg_search_time']*1000:.2f}ms, recall@{k}={recall:.3f}")
            print(f"  ✅ NumPy: search={numpy_result['avg_search_time']*1000:.2f}ms")
            print(f"  🚀 Speedup: {numpy_result['avg_search_time']/faiss_result['avg_search_time']:.1f}x")

# === EXECUTAR BENCHMARK ===

benchmark = VectorStoreComparison(dimension=384)

# Inicializar estrutura de resultados
for method in ['numpy', 'faiss']:
    benchmark.results[method] = {
        'sizes': [],
        'build_times': [],
        'search_times': [],
        'total_times': []
    }
benchmark.results['faiss']['recalls'] = []

# Run benchmark
benchmark.run_comparison(sizes=[1000, 5000, 10000, 25000])

# === VISUALIZAÇÃO ===

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Search time comparison
ax = axes[0, 0]
sizes = benchmark.results['numpy']['sizes']
numpy_times = np.array(benchmark.results['numpy']['search_times']) * 1000  # to ms
faiss_times = np.array(benchmark.results['faiss']['search_times']) * 1000

ax.plot(sizes, numpy_times, marker='o', label='NumPy Brute-Force',
        linewidth=2, markersize=8, color='coral')
ax.plot(sizes, faiss_times, marker='s', label='FAISS',
        linewidth=2, markersize=8, color='skyblue')
ax.set_xlabel('Database Size (vectors)')
ax.set_ylabel('Avg Search Time (ms)')
ax.set_title('Search Latency: FAISS vs NumPy')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
ax.grid(alpha=0.3)

# 2. Build time
ax = axes[0, 1]
faiss_build_times = np.array(benchmark.results['faiss']['build_times'])

ax.bar(range(len(sizes)), faiss_build_times, color='lightgreen', alpha=0.7)
ax.set_xlabel('Database Size Index')
ax.set_ylabel('Build Time (s)')
ax.set_title('FAISS Index Build Time')
ax.set_xticks(range(len(sizes)))
ax.set_xticklabels([f'{s:,}' for s in sizes], rotation=45)
ax.grid(axis='y', alpha=0.3)

# 3. Speedup factor
ax = axes[1, 0]
speedups = numpy_times / faiss_times

ax.bar(range(len(sizes)), speedups, color='purple', alpha=0.7)
ax.set_xlabel('Database Size')
ax.set_ylabel('Speedup Factor (x)')
ax.set_title('FAISS Speedup over NumPy')
ax.set_xticks(range(len(sizes)))
ax.set_xticklabels([f'{s:,}' for s in sizes], rotation=45)
ax.axhline(1, color='red', linestyle='--', label='Baseline')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 4. Recall@k
ax = axes[1, 1]
recalls = benchmark.results['faiss']['recalls']

ax.plot(sizes, recalls, marker='o', linewidth=2, markersize=8,
        color='green', label='FAISS Recall@5')
ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Perfect (1.0)')
ax.set_xlabel('Database Size (vectors)')
ax.set_ylabel('Recall@5')
ax.set_title('FAISS Accuracy (vs Exact Search)')
ax.set_ylim([0.95, 1.01])
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('vector_store_comparison.png', dpi=150, bbox_inches='tight')
print("\n📊 Gráfico salvo: vector_store_comparison.png")

# === SUMMARY TABLE ===

print("\n" + "="*70)
print("📊 TABELA COMPARATIVA FINAL")
print("="*70)
print(f"\n{'Size':<10} {'Method':<10} {'Build(s)':<10} {'Search(ms)':<12} {'Speedup':<10} {'Recall@5':<10}")
print("-"*70)

for i, size in enumerate(sizes):
    numpy_time = benchmark.results['numpy']['search_times'][i] * 1000
    faiss_time = benchmark.results['faiss']['search_times'][i] * 1000
    faiss_build = benchmark.results['faiss']['build_times'][i]
    speedup = numpy_time / faiss_time
    recall = benchmark.results['faiss']['recalls'][i]

    print(f"{size:<10,} {'NumPy':<10} {'-':<10} {numpy_time:<12.2f} {'-':<10} {'-':<10}")
    print(f"{'':<10} {'FAISS':<10} {faiss_build:<10.4f} {faiss_time:<12.2f} {speedup:<10.1f}x {recall:<10.3f}")
    print()

print("\n💡 RECOMENDAÇÕES:")
print("-"*70)
print("✅ FAISS: Melhor para produção com grandes volumes (>10K docs)")
print("✅ NumPy: OK para protótipos pequenos (<1K docs)")
print("✅ ChromaDB: Melhor developer experience, persistência fácil")
print("✅ FAISS + ChromaDB: Combinar ambos para melhor resultado")
print("\n🔗 FAISS Advanced:")
print("  - IndexIVFFlat: Inverted file index (mais rápido, aprox.)")
print("  - IndexHNSW: Hierarchical NSW (melhor recall)")
  - IndexPQ: Product Quantization (menor memória)")
