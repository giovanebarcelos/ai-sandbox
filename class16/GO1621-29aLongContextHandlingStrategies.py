# GO1621-29aLongContextHandlingStrategies
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import re

class LongContextManager:
    """
    Estratégias para lidar com contextos longos

    Methods:
    1. Chunking: dividir em partes
    2. Sliding window: janela deslizante
    3. Sparse attention: atenção esparsa
    4. Hierarchical: processar níveis
    5. Compression: comprimir contexto
    """

    def __init__(self, max_context_length=4096):
        self.max_context_length = max_context_length

    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Divide texto em chunks com overlap

        Overlap mantém contexto entre chunks
        """
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)

            if i + chunk_size >= len(words):
                break

        return chunks

    def sliding_window(self, tokens: List[int], window_size: int = 2048, stride: int = 1024) -> List[List[int]]:
        """
        Sliding window sobre tokens

        Useful for: classification, embedding generation
        """
        windows = []

        for i in range(0, len(tokens), stride):
            window = tokens[i:i + window_size]

            if len(window) < window_size and windows:
                # Last window, skip if too small
                break

            windows.append(window)

        return windows

    def hierarchical_summarize(self, text: str, levels: int = 2) -> Dict:
        """
        Hierarchical summarization for long documents

        Level 0: Original text
        Level 1: Summarize in chunks
        Level 2: Summarize summaries
        """
        results = {'level_0': text}

        current_text = text

        for level in range(1, levels + 1):
            # Chunk current text
            chunks = self.chunk_text(current_text, chunk_size=1000, overlap=100)

            # Simulate summarization (in real: use model)
            summaries = []
            for chunk in chunks:
                # Extract first sentence as summary (simplified)
                sentences = chunk.split('.')
                summary = sentences[0][:200] if sentences else chunk[:200]
                summaries.append(summary)

            # Combine summaries
            current_text = ' '.join(summaries)
            results[f'level_{level}'] = current_text

            # Early stop if text is short enough
            if len(current_text.split()) < 500:
                break

        return results

    def sparse_attention_pattern(self, seq_length: int, pattern: str = 'fixed') -> np.ndarray:
        """
        Generate sparse attention pattern

        Patterns:
        - fixed: attend to fixed positions (e.g., every 64th)
        - local: attend to local window
        - strided: attend with stride
        - global: few global tokens attend to all
        """
        attention_matrix = np.zeros((seq_length, seq_length))

        if pattern == 'fixed':
            # Attend to every 64th position + local
            for i in range(seq_length):
                # Local window (±16)
                attention_matrix[i, max(0, i-16):min(seq_length, i+17)] = 1

                # Global tokens (every 64th)
                for j in range(0, seq_length, 64):
                    attention_matrix[i, j] = 1

        elif pattern == 'local':
            # Local window only
            window = 32
            for i in range(seq_length):
                attention_matrix[i, max(0, i-window):min(seq_length, i+window+1)] = 1

        elif pattern == 'strided':
            # Attend with stride
            stride = 8
            for i in range(seq_length):
                # Local
                attention_matrix[i, max(0, i-8):min(seq_length, i+9)] = 1
                # Strided
                attention_matrix[i, i::stride] = 1

        return attention_matrix

    def compress_context(self, text: str, target_length: int = 1000) -> str:
        """
        Compress long text to fit context window

        Methods:
        - Extract key sentences
        - Remove redundancy
        - Prioritize important info
        """
        sentences = text.split('.')

        # Score sentences by importance (simplified)
        # In real system: use TF-IDF, embedding similarity, etc.
        sentence_scores = []

        important_keywords = ['important', 'key', 'critical', 'main', 'summary', 
                             'conclusion', 'result', 'finding']

        for sent in sentences:
            score = sum(1 for kw in important_keywords if kw in sent.lower())
            # Longer sentences get slight bonus
            score += len(sent.split()) / 100
            sentence_scores.append((sent, score))

        # Sort by score
        sentence_scores.sort(key=lambda x: x[1], reverse=True)

        # Take top sentences until target length
        compressed = []
        current_length = 0

        for sent, score in sentence_scores:
            sent_len = len(sent.split())
            if current_length + sent_len <= target_length:
                compressed.append(sent)
                current_length += sent_len
            else:
                break

        return '. '.join(compressed)

# === DEMO ===

print("📏 Long Context Handling Strategies\n")
print("="*70)

manager = LongContextManager(max_context_length=4096)

# Generate long text
long_text = "Artificial intelligence has revolutionized technology. " * 500  # ~2500 words

print(f"Long text: {len(long_text.split())} words\n")

# Strategy 1: Chunking
print("📌 Strategy 1: Chunking with Overlap\n")

chunks = manager.chunk_text(long_text, chunk_size=100, overlap=20)

print(f"   Total chunks: {len(chunks)}")
print(f"   Chunk size: ~100 words")
print(f"   Overlap: 20 words")
print(f"   Coverage: {sum(len(c.split()) for c in chunks)} words\n")

# Strategy 2: Hierarchical summarization
print("📌 Strategy 2: Hierarchical Summarization\n")

hierarchy = manager.hierarchical_summarize(long_text[:10000], levels=3)

for level, text in hierarchy.items():
    word_count = len(text.split())
    print(f"   {level}: {word_count} words")

print()

# Strategy 3: Sparse attention
print("📌 Strategy 3: Sparse Attention Patterns\n")

seq_len = 256

patterns = ['fixed', 'local', 'strided']

attention_matrices = {}
sparsity_ratios = {}

for pattern in patterns:
    attn_matrix = manager.sparse_attention_pattern(seq_len, pattern)
    attention_matrices[pattern] = attn_matrix

    sparsity = 1 - (np.sum(attn_matrix) / (seq_len ** 2))
    sparsity_ratios[pattern] = sparsity

    print(f"   {pattern.capitalize()} attention:")
    print(f"      Sparsity: {sparsity:.1%}")
    print(f"      Memory reduction: {sparsity:.1%}")

print()

# Strategy 4: Compression
print("📌 Strategy 4: Context Compression\n")

compressed = manager.compress_context(long_text, target_length=200)

original_words = len(long_text.split())
compressed_words = len(compressed.split())
compression_ratio = original_words / compressed_words

print(f"   Original: {original_words} words")
print(f"   Compressed: {compressed_words} words")
print(f"   Compression: {compression_ratio:.1f}x\n")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Sparse attention patterns
for idx, (pattern, attn_matrix) in enumerate(attention_matrices.items()):
    if idx < 3:
        if idx == 0:
            ax = axes[0, 0]
        elif idx == 1:
            ax = axes[0, 1]
        else:
            ax = axes[1, 0]

        im = ax.imshow(attn_matrix, cmap='Blues', aspect='auto')
        ax.set_title(f'{pattern.capitalize()} Attention\n(Sparsity: {sparsity_ratios[pattern]:.0%})')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax)

# 2. Context length evolution
ax = axes[1, 1]

models = ['GPT-2\n(2019)', 'GPT-3\n(2020)', 'GPT-4-32K\n(2023)', 'Claude 3\n(2024)', 'Gemini 1.5\n(2024)']
context_lengths = [1024, 4096, 32768, 200000, 1000000]

bars = ax.bar(models, context_lengths, color=['red', 'orange', 'yellow', 'lightgreen', 'green'], alpha=0.7)
ax.set_ylabel('Context Length (tokens)')
ax.set_yscale('log')
ax.set_title('LLM Context Length Evolution')
ax.grid(axis='y', alpha=0.3)

for bar, length in zip(bars, context_lengths):
    height = bar.get_height()
    if length < 10000:
        label = f'{length:,}'
    elif length < 100000:
        label = f'{length//1000}K'
    else:
        label = f'{length//1000}K'

    ax.text(bar.get_x() + bar.get_width()/2., height * 1.1,
            label, ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('long_context_handling.png', dpi=150, bbox_inches='tight')
print("📊 Gráfico salvo: long_context_handling.png")

print("\n✅ Long context strategies implementado!")
print("\n💡 BEST PRACTICES:")
print("   - Chunking: Use overlap to preserve context")
print("   - Hierarchical: Good for documents > 100K tokens")
print("   - Sparse attention: 90% memory reduction")
print("   - Compression: Extract key information")
print("   - Consider retrieval (RAG) for very long contexts")
print("\n💡 WHEN TO USE EACH:")
print("   Chunking: Classification, search")
print("   Hierarchical: Summarization, question answering")
print("   Sparse attention: Training long-context models")
print("   Compression: Fixed context window constraints")
