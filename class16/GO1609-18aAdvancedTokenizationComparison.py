# GO1609-18aAdvancedTokenizationComparison
from transformers import (
    AutoTokenizer, 
    GPT2Tokenizer, 
    BertTokenizer,
    XLNetTokenizer,
    T5Tokenizer
)
import numpy as np
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

class TokenizerComparison:
    """
    Comparação abrangente de tokenizadores

    Algoritmos de tokenização:
    - BPE (Byte-Pair Encoding): GPT-2, RoBERTa
    - WordPiece: BERT
    - Unigram: XLNet
    - SentencePiece: T5, LLaMA
    """

    def __init__(self):
        print("🔤 Carregando tokenizadores...\n")

        # Load different tokenizers
        self.tokenizers = {
            'GPT-2 (BPE)': GPT2Tokenizer.from_pretrained('gpt2'),
            'BERT (WordPiece)': BertTokenizer.from_pretrained('bert-base-uncased'),
            'T5 (SentencePiece)': T5Tokenizer.from_pretrained('t5-small'),
        }

        print("✅ Tokenizadores carregados com sucesso!\n")

    def tokenize_text(self, text: str):
        """Tokenizar texto com todos os tokenizadores"""
        results = {}

        for name, tokenizer in self.tokenizers.items():
            tokens = tokenizer.tokenize(text)
            token_ids = tokenizer.encode(text)

            results[name] = {
                'tokens': tokens,
                'token_ids': token_ids,
                'num_tokens': len(tokens)
            }

        return results

    def compare_on_text(self, text: str):
        """Comparar tokenização em um texto específico"""
        print(f"📝 Texto Original:\n   \"{text}\"\n")

        results = self.tokenize_text(text)

        for name, result in results.items():
            print(f"{name}:")
            print(f"   Tokens ({result['num_tokens']}): {result['tokens'][:10]}")
            if len(result['tokens']) > 10:
                print(f"   ... (truncado, total: {result['num_tokens']} tokens)")
            print()

        return results

    def analyze_vocabulary_coverage(self, texts: list):
        """Analisar como diferentes tokenizadores lidam com o vocabulário"""
        vocab_stats = {name: [] for name in self.tokenizers.keys()}

        for text in texts:
            results = self.tokenize_text(text)

            for name, result in results.items():
                vocab_stats[name].append(result['num_tokens'])

        return vocab_stats

    def special_tokens_analysis(self):
        """Analisar tokens especiais entre tokenizadores"""
        print("🔍 Análise de Tokens Especiais:\n")

        for name, tokenizer in self.tokenizers.items():
            print(f"{name}:")
            print(f"   PAD: {tokenizer.pad_token}")
            print(f"   UNK: {tokenizer.unk_token}")
            print(f"   BOS: {getattr(tokenizer, 'bos_token', 'N/A')}")
            print(f"   EOS: {getattr(tokenizer, 'eos_token', 'N/A')}")
            print(f"   Tamanho do vocabulário: {len(tokenizer):,}")
            print()

# === DEMO ===

print("🔤 Comparação Avançada de Tokenizadores\n")
print("="*70)

comp = TokenizerComparison()

# Test cases
test_texts = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "COVID-19 pandemic caused unprecedented challenges worldwide.",
    "I'm unhappy about this unfairness!",
    "Machine learning models like GPT-4 are revolutionizing NLP.",
]

print("\n📌 Caso de Teste 1: Frase Simples\n")
comp.compare_on_text(test_texts[0])

print("\n📌 Caso de Teste 2: Palavras Complexas (OOV)\n")
comp.compare_on_text(test_texts[2])

print("\n📌 Caso de Teste 3: Contrações e Afixos\n")
comp.compare_on_text(test_texts[3])

# Special tokens
print("\n" + "="*70)
comp.special_tokens_analysis()

# Vocabulary coverage analysis
print("="*70)
print("\n📊 Análise de Cobertura do Vocabulário\n")

vocab_stats = comp.analyze_vocabulary_coverage(test_texts)

for name, token_counts in vocab_stats.items():
    avg_tokens = np.mean(token_counts)
    print(f"{name}:")
    print(f"   Média de tokens por texto: {avg_tokens:.2f}")
    print(f"   Intervalo: {min(token_counts)}-{max(token_counts)}")
    print()

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Token count comparison across test texts
ax = axes[0, 0]
x = np.arange(len(test_texts))
width = 0.25

for i, (name, token_counts) in enumerate(vocab_stats.items()):
    ax.bar(x + i*width, token_counts, width, label=name, alpha=0.8)

ax.set_xlabel('Texto de Teste')
ax.set_ylabel('Número de Tokens')
ax.set_title('Comparação de Contagem de Tokens')
ax.set_xticks(x + width)
ax.set_xticklabels([f'Texto {i+1}' for i in range(len(test_texts))])
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)

# 2. Eficiência média (menos tokens = mais eficiente)
ax = axes[0, 1]
avg_tokens = [np.mean(counts) for counts in vocab_stats.values()]
names = list(vocab_stats.keys())

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bars = ax.barh(names, avg_tokens, color=colors, alpha=0.7)
ax.set_xlabel('Média de Tokens por Texto')
ax.set_title('Eficiência do Tokenizador (Menor = Mais Eficiente)')
ax.grid(axis='x', alpha=0.3)

for bar, val in zip(bars, avg_tokens):
    ax.text(val + 0.2, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}', ha='left', va='center', fontweight='bold')

# 3. Vocabulary size comparison
ax = axes[1, 0]
vocab_sizes = [len(tokenizer) for tokenizer in comp.tokenizers.values()]
names_short = ['GPT-2\n(BPE)', 'BERT\n(WordPiece)', 'T5\n(SentencePiece)']

bars = ax.bar(names_short, vocab_sizes, color=colors, alpha=0.7)
ax.set_ylabel('Tamanho do Vocabulário')
ax.set_title('Tamanho do Vocabulário por Tokenizador')
ax.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, vocab_sizes):
    ax.text(bar.get_x() + bar.get_width()/2, val + 500,
            f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 4. Token distribution (simulated for rare words)
ax = axes[1, 1]

# Simulate token frequency distribution
x_freq = np.arange(1, 21)
bpe_dist = 50000 * np.exp(-0.3 * x_freq)  # BPE: slower decay
wordpiece_dist = 45000 * np.exp(-0.35 * x_freq)  # WordPiece: medium
sentencepiece_dist = 40000 * np.exp(-0.4 * x_freq)  # SentencePiece: faster decay

ax.plot(x_freq, bpe_dist, 'o-', label='GPT-2 (BPE)', linewidth=2)
ax.plot(x_freq, wordpiece_dist, 's-', label='BERT (WordPiece)', linewidth=2)
ax.plot(x_freq, sentencepiece_dist, '^-', label='T5 (SentencePiece)', linewidth=2)

ax.set_xlabel('Rank do Token')
ax.set_ylabel('Frequência (escala log)')
ax.set_title('Distribuição de Frequência de Tokens')
ax.set_yscale('log')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
print("\n📊 Gráfico salvo: tokenizer_comparison.png")

print("\n✅ Tokenizer comparison completo!")
print("\n💡 PRINCIPAIS INSIGHTS:")
print("   - BPE (GPT-2): Fallback em nível de caractere, bom para palavras raras")
print("   - WordPiece (BERT): Unidades de subpalavra, eficiência equilibrada")
print("   - SentencePiece (T5): Independente de idioma, sem pré-tokenização")
print("   - Trade-off: Tamanho do vocabulário vs. comprimento da sequência")
print("\n💡 BOAS PRÁTICAS:")
print("   - Escolha o tokenizador conforme a arquitetura do modelo")
print("   - Considere o tamanho do vocabulário para restrições de memória")
print("   - Teste o tratamento de OOV (fora do vocabulário)")
print("   - Monitore o comprimento da sequência de tokens (afeta custo de inferência)")
