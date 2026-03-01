# GO1705-16bAdvancedDocumentChunkingStrategies
import re
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class AdvancedChunker:
    """
    Estratégias avançadas de chunking:
    1. Semantic chunking (por tópicos)
    2. Sentence-window (sentenças + contexto)
    3. Recursive hierarchical (hierárquico)
    4. Sliding window com overlap inteligente
    5. Document-aware (preserva estrutura)
    """

    def __init__(self):
        self.chunking_stats = []

    def sentence_tokenize(self, text: str) -> List[str]:
        """Divide texto em sentenças"""
        # Simple sentence tokenizer
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def fixed_size_chunking(self, text: str, chunk_size: int = 500, 
                           overlap: int = 50) -> List[Dict]:
        """Chunking de tamanho fixo com overlap"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) < 20:  # Skip very small chunks
                continue

            chunk_text = ' '.join(chunk_words)
            chunks.append({
                'text': chunk_text,
                'method': 'fixed_size',
                'start_word': i,
                'size': len(chunk_words),
                'overlap': overlap if i > 0 else 0
            })

        return chunks

    def sentence_window_chunking(self, text: str, sentences_per_chunk: int = 5,
                                window_size: int = 2) -> List[Dict]:
        """
        Chunking por sentenças com janela de contexto

        Cada chunk contém N sentenças principais + contexto anterior/posterior
        """
        sentences = self.sentence_tokenize(text)
        chunks = []

        for i in range(0, len(sentences), sentences_per_chunk):
            # Main sentences
            main_start = i
            main_end = min(i + sentences_per_chunk, len(sentences))

            # Add context window
            context_start = max(0, main_start - window_size)
            context_end = min(len(sentences), main_end + window_size)

            # Build chunk with context
            context_before = sentences[context_start:main_start]
            main_sentences = sentences[main_start:main_end]
            context_after = sentences[main_end:context_end]

            chunk_text = ' '.join(context_before + main_sentences + context_after)

            chunks.append({
                'text': chunk_text,
                'method': 'sentence_window',
                'main_sentences': main_sentences,
                'context_before': len(context_before),
                'context_after': len(context_after),
                'sentence_index': i
            })

        return chunks

    def semantic_chunking(self, text: str, similarity_threshold: float = 0.7) -> List[Dict]:
        """
        Chunking semântico: agrupa sentenças similares

        Usa similaridade entre sentenças para determinar limites
        """
        sentences = self.sentence_tokenize(text)

        if len(sentences) < 2:
            return [{'text': text, 'method': 'semantic', 'sentences': len(sentences)}]

        # Simple semantic grouping (word overlap)
        chunks = []
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            # Calculate word overlap with current chunk
            current_words = set(' '.join(current_chunk).lower().split())
            next_words = set(sentences[i].lower().split())

            if len(current_words) == 0:
                similarity = 0
            else:
                overlap = len(current_words & next_words)
                similarity = overlap / len(current_words)

            if similarity >= similarity_threshold or len(current_chunk) < 3:
                current_chunk.append(sentences[i])
            else:
                # Start new chunk
                chunks.append({
                    'text': ' '.join(current_chunk),
                    'method': 'semantic',
                    'sentences': len(current_chunk),
                    'similarity': similarity
                })
                current_chunk = [sentences[i]]

        # Add last chunk
        if current_chunk:
            chunks.append({
                'text': ' '.join(current_chunk),
                'method': 'semantic',
                'sentences': len(current_chunk)
            })

        return chunks

    def hierarchical_chunking(self, text: str, levels: int = 3) -> List[Dict]:
        """
        Chunking hierárquico: múltiplos níveis de granularidade

        Level 1: Documento inteiro (summary)
        Level 2: Seções grandes
        Level 3: Parágrafos/chunks pequenos
        """
        chunks = []

        # Level 1: Full document
        chunks.append({
            'text': text[:500] + '...',  # Summary
            'method': 'hierarchical',
            'level': 1,
            'type': 'document',
            'full_text': text
        })

        # Level 2: Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        for i, para in enumerate(paragraphs):
            if len(para) > 100:  # Only substantial paragraphs
                chunks.append({
                    'text': para,
                    'method': 'hierarchical',
                    'level': 2,
                    'type': 'paragraph',
                    'index': i,
                    'parent': 0  # Links to level 1
                })

        # Level 3: Further split large paragraphs
        chunk_id = len(chunks)
        for i, para in enumerate(paragraphs):
            if len(para.split()) > 200:  # Large paragraphs
                sentences = self.sentence_tokenize(para)
                for j in range(0, len(sentences), 3):
                    chunk_text = ' '.join(sentences[j:j+3])
                    chunks.append({
                        'text': chunk_text,
                        'method': 'hierarchical',
                        'level': 3,
                        'type': 'sub_paragraph',
                        'parent': i + 1  # Links to level 2
                    })

        return chunks

    def document_aware_chunking(self, text: str) -> List[Dict]:
        """
        Chunking que preserva estrutura do documento
        (headers, listas, code blocks, etc.)
        """
        chunks = []

        # Detect markdown headers
        lines = text.split('\n')
        current_section = {'title': 'Introduction', 'content': []}

        for line in lines:
            # Check for header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # Save previous section
                if current_section['content']:
                    chunks.append({
                        'text': '\n'.join(current_section['content']),
                        'method': 'document_aware',
                        'section_title': current_section['title'],
                        'has_header': True
                    })

                # Start new section
                current_section = {
                    'title': header_match.group(2),
                    'content': []
                }
            else:
                current_section['content'].append(line)

        # Add last section
        if current_section['content']:
            chunks.append({
                'text': '\n'.join(current_section['content']),
                'method': 'document_aware',
                'section_title': current_section['title'],
                'has_header': True
            })

        return chunks

    def compare_strategies(self, text: str) -> Dict:
        """Compara todas as estratégias de chunking"""
        results = {
            'fixed_size': self.fixed_size_chunking(text, chunk_size=100, overlap=20),
            'sentence_window': self.sentence_window_chunking(text, sentences_per_chunk=3, window_size=1),
            'semantic': self.semantic_chunking(text, similarity_threshold=0.5),
            'hierarchical': self.hierarchical_chunking(text),
            'document_aware': self.document_aware_chunking(text)
        }

        return results

# === EXEMPLO DE USO ===

print("\n📚 Advanced Document Chunking Demo\n")
print("="*70)

# Sample document (markdown with structure)
sample_doc = """# Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence. It enables computers to learn from data without being explicitly programmed.

## Types of Learning

There are three main types of machine learning approaches.

### Supervised Learning

In supervised learning, we train models on labeled data. The algorithm learns to map inputs to outputs. Common algorithms include linear regression, decision trees, and neural networks.

### Unsupervised Learning

Unsupervised learning works with unlabeled data. The goal is to discover patterns. Clustering and dimensionality reduction are key techniques.

### Reinforcement Learning

Reinforcement learning involves agents learning through interaction. They receive rewards or penalties for actions. This approach is used in robotics and game playing.

## Applications

Machine learning powers many modern applications. It's used in recommendation systems, computer vision, natural language processing, and autonomous vehicles.

## Conclusion

Understanding these fundamentals is crucial for anyone working in AI. The field continues to evolve rapidly with new techniques emerging regularly.
"""

chunker = AdvancedChunker()

# Compare all strategies
results = chunker.compare_strategies(sample_doc)

print("\n📊 COMPARAÇÃO DE ESTRATÉGIAS\n")
print("-"*70)

for strategy, chunks in results.items():
    print(f"\n{strategy.upper()}:")
    print(f"  Total chunks: {len(chunks)}")

    if chunks:
        sizes = [len(c['text'].split()) for c in chunks]
        print(f"  Avg size: {np.mean(sizes):.0f} words")
        print(f"  Size range: {min(sizes)}-{max(sizes)} words")

        # Show first chunk
        print(f"  First chunk preview: {chunks[0]['text'][:80]}...")

# === VISUALIZAÇÃO ===

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Number of chunks per strategy
ax = axes[0, 0]
strategies = list(results.keys())
chunk_counts = [len(results[s]) for s in strategies]

ax.bar(range(len(strategies)), chunk_counts, color='skyblue', alpha=0.7)
ax.set_xticks(range(len(strategies)))
ax.set_xticklabels(strategies, rotation=45, ha='right')
ax.set_ylabel('Number of Chunks')
ax.set_title('Chunks por Estratégia')
ax.grid(axis='y', alpha=0.3)

# 2. Average chunk size
ax = axes[0, 1]
avg_sizes = [np.mean([len(c['text'].split()) for c in results[s]]) 
             for s in strategies]

ax.barh(strategies, avg_sizes, color='lightgreen', alpha=0.7)
ax.set_xlabel('Average Words per Chunk')
ax.set_title('Tamanho Médio dos Chunks')
ax.grid(axis='x', alpha=0.3)

# 3. Size distribution (box plot)
ax = axes[1, 0]
size_data = [[len(c['text'].split()) for c in results[s]] for s in strategies]

ax.boxplot(size_data, labels=strategies)
ax.set_xticklabels(strategies, rotation=45, ha='right')
ax.set_ylabel('Words per Chunk')
ax.set_title('Distribuição de Tamanhos')
ax.grid(axis='y', alpha=0.3)

# 4. Chunk size histogram for one strategy
ax = axes[1, 1]
fixed_sizes = [len(c['text'].split()) for c in results['fixed_size']]
semantic_sizes = [len(c['text'].split()) for c in results['semantic']]

ax.hist([fixed_sizes, semantic_sizes], label=['Fixed Size', 'Semantic'],
        bins=15, alpha=0.6)
ax.set_xlabel('Words per Chunk')
ax.set_ylabel('Frequency')
ax.set_title('Comparação: Fixed vs Semantic')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('chunking_strategies_comparison.png', dpi=150, bbox_inches='tight')
print("\n📊 Gráfico salvo: chunking_strategies_comparison.png")

# Recommendations
print("\n💡 RECOMENDAÇÕES:")
print("="*70)
print("✅ Fixed Size: Rápido, consistente, boa baseline")
print("✅ Sentence Window: Melhor contexto, bom para Q&A")
print("✅ Semantic: Preserva coerência semântica")
print("✅ Hierarchical: Permite busca multi-nível")
print("✅ Document-Aware: Melhor para docs estruturados (MD, HTML)")
print("\n🎯 Escolha baseado em:")
print("  - Tipo de documento (estruturado vs não estruturado)")
print("  - Tipo de query (específica vs exploratória)")
print("  - Trade-off precisão vs recall")

print("\n✅ Advanced Chunking implementado!")
