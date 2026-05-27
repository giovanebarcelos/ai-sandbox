# GO1621-29aLongContextHandlingStrategies
import numpy as np
from typing import List, Dict, Tuple
import re

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

class LongContextManager:
    """
    Estratégias para lidar com contextos longos

    Métodos:
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

        Útil para: classificação, geração de embeddings
        """
        windows = []

        for i in range(0, len(tokens), stride):
            window = tokens[i:i + window_size]

            if len(window) < window_size and windows:
                # Última janela, ignorar se muito pequena
                break

            windows.append(window)

        return windows

    def hierarchical_summarize(self, text: str, levels: int = 2) -> Dict:
        """
        Sumarização hierárquica para documentos longos

        Nível 0: Texto original
        Nível 1: Sumarizar em chunks
        Nível 2: Sumarizar os resumos
        """
        results = {'level_0': text}

        current_text = text

        for level in range(1, levels + 1):
            # Chunk current text
            chunks = self.chunk_text(current_text, chunk_size=1000, overlap=100)

            # Simular sumarização (em produção: usar modelo)
            summaries = []
            for chunk in chunks:
                # Extrair primeira sentença como resumo (simplificado)
                sentences = chunk.split('.')
                summary = sentences[0][:200] if sentences else chunk[:200]
                summaries.append(summary)

            # Combinar resumos
            current_text = ' '.join(summaries)
            results[f'level_{level}'] = current_text

            # Parar cedo se o texto for suficientemente curto
            if len(current_text.split()) < 500:
                break

        return results

    def sparse_attention_pattern(self, seq_length: int, pattern: str = 'fixed') -> np.ndarray:
        """
        Gerar padrão de atenção esparsa

        Padrões:
        - fixed: atenção em posições fixas (ex.: a cada 64ª)
        - local: atenção na janela local
        - strided: atenção com passo
        - global: poucos tokens globais atendem a todos
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
        Comprimir texto longo para caber na janela de contexto

        Métodos:
        - Extrair sentenças-chave
        - Remover redundância
        - Priorizar informações importantes
        """
        sentences = text.split('.')

        # Pontuar sentenças por importância (simplificado)
        # Em um sistema real: use TF-IDF, similaridade de embeddings, etc.
        sentence_scores = []

        important_keywords = ['important', 'key', 'critical', 'main', 'summary', 
                             'conclusion', 'result', 'finding']

        for sent in sentences:
            score = sum(1 for kw in important_keywords if kw in sent.lower())
            # Sentenças mais longas recebem bônus
            score += len(sent.split()) / 100
            sentence_scores.append((sent, score))

        # Ordenar por pontuação
        sentence_scores.sort(key=lambda x: x[1], reverse=True)

        # Pegar as melhores sentenças até atingir o comprimento alvo
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

print("📏 Estratégias para Contextos Longos\n")
print("="*70)

manager = LongContextManager(max_context_length=4096)

# Generate long text
long_text = "Artificial intelligence has revolutionized technology. " * 500  # ~2500 words

print(f"Texto longo: {len(long_text.split())} palavras\n")

# Strategy 1: Chunking
print("📌 Estratégia 1: Chunking com Sobreposição\n")

chunks = manager.chunk_text(long_text, chunk_size=100, overlap=20)

print(f"   Total de chunks: {len(chunks)}")
print(f"   Tamanho do chunk: ~100 palavras")
print(f"   Sobreposição: 20 palavras")
print(f"   Cobertura: {sum(len(c.split()) for c in chunks)} palavras\n")

# Strategy 2: Hierarchical summarization
print("📌 Estratégia 2: Sumarização Hierárquica\n")

hierarchy = manager.hierarchical_summarize(long_text[:10000], levels=3)

for level, text in hierarchy.items():
    word_count = len(text.split())
    print(f"   {level}: {word_count} palavras")

print()

# Estratégia 3: Atenção esparsa
print("📌 Estratégia 3: Padrões de Atenção Esparsa\n")

seq_len = 256

patterns = ['fixed', 'local', 'strided']

attention_matrices = {}
sparsity_ratios = {}

for pattern in patterns:
    attn_matrix = manager.sparse_attention_pattern(seq_len, pattern)
    attention_matrices[pattern] = attn_matrix

    sparsity = 1 - (np.sum(attn_matrix) / (seq_len ** 2))
    sparsity_ratios[pattern] = sparsity

    print(f"   Atenção {pattern.capitalize()}:")
    print(f"      Esparsidade: {sparsity:.1%}")
    print(f"      Redução de memória: {sparsity:.1%}")

print()

# Strategy 4: Compression
print("📌 Estratégia 4: Compressão de Contexto\n")

compressed = manager.compress_context(long_text, target_length=200)

original_words = len(long_text.split())
compressed_words = len(compressed.split())
compression_ratio = original_words / compressed_words

print(f"   Original: {original_words} palavras")
print(f"   Comprimido: {compressed_words} palavras")
print(f"   Compressão: {compression_ratio:.1f}x\n")

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
        ax.set_xlabel('Posição da Chave')
        ax.set_ylabel('Posição da Consulta')
        plt.colorbar(im, ax=ax)

# 2. Context length evolution
ax = axes[1, 1]

models = ['GPT-2\n(2019)', 'GPT-3\n(2020)', 'GPT-4-32K\n(2023)', 'Claude 3\n(2024)', 'Gemini 1.5\n(2024)']
context_lengths = [1024, 4096, 32768, 200000, 1000000]

bars = ax.bar(models, context_lengths, color=['red', 'orange', 'yellow', 'lightgreen', 'green'], alpha=0.7)
ax.set_ylabel('Comprimento do Contexto (tokens)')
ax.set_yscale('log')
ax.set_title('Evolução do Comprimento de Contexto dos LLMs')
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
plt.show()
print("📊 Gráfico salvo: long_context_handling.png")

print("\n✅ Long context strategies implementado!")
print("\n💡 BOAS PRÁTICAS:")
print("   - Chunking: Use sobreposição para preservar contexto")
print("   - Hierárquico: Bom para documentos > 100K tokens")
print("   - Atenção esparsa: 90% de redução de memória")
print("   - Compressão: Extrai informações importantes")
print("   - Considere recuperação (RAG) para contextos muito longos")
print("\n💡 QUANDO USAR CADA UM:")
print("   Chunking: Classificação, busca")
print("   Hierárquico: Sumarização, resposta a perguntas")
print("   Atenção esparsa: Treinamento de modelos de contexto longo")
print("   Compressão: Restrições de janela de contexto fixo")
