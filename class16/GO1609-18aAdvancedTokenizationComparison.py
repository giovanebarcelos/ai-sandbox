# GO1609-18aAdvancedTokenizationComparison
from transformers import (
    AutoTokenizer, 
    GPT2Tokenizer, 
    BertTokenizer,
    XLNetTokenizer,
    T5Tokenizer
)
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

class TokenizerComparison:
    """
    Comprehensive tokenizer comparison

    Tokenization algorithms:
    - BPE (Byte-Pair Encoding): GPT-2, RoBERTa
    - WordPiece: BERT
    - Unigram: XLNet
    - SentencePiece: T5, LLaMA
    """

    def __init__(self):
        print("🔤 Loading tokenizers...\n")

        # Load different tokenizers
        self.tokenizers = {
            'GPT-2 (BPE)': GPT2Tokenizer.from_pretrained('gpt2'),
            'BERT (WordPiece)': BertTokenizer.from_pretrained('bert-base-uncased'),
            'T5 (SentencePiece)': T5Tokenizer.from_pretrained('t5-small'),
        }

        print("✅ Tokenizers loaded successfully!\n")

    def tokenize_text(self, text: str):
        """Tokenize text with all tokenizers"""
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
        """Compare tokenization on a specific text"""
        print(f"📝 Original Text:\n   \"{text}\"\n")

        results = self.tokenize_text(text)

        for name, result in results.items():
            print(f"{name}:")
            print(f"   Tokens ({result['num_tokens']}): {result['tokens'][:10]}")
            if len(result['tokens']) > 10:
                print(f"   ... (truncated, total: {result['num_tokens']} tokens)")
            print()

        return results

    def analyze_vocabulary_coverage(self, texts: list):
        """Analyze how different tokenizers handle vocabulary"""
        vocab_stats = {name: [] for name in self.tokenizers.keys()}

        for text in texts:
            results = self.tokenize_text(text)

            for name, result in results.items():
                vocab_stats[name].append(result['num_tokens'])

        return vocab_stats

    def special_tokens_analysis(self):
        """Analyze special tokens across tokenizers"""
        print("🔍 Special Tokens Analysis:\n")

        for name, tokenizer in self.tokenizers.items():
            print(f"{name}:")
            print(f"   PAD: {tokenizer.pad_token}")
            print(f"   UNK: {tokenizer.unk_token}")
            print(f"   BOS: {getattr(tokenizer, 'bos_token', 'N/A')}")
            print(f"   EOS: {getattr(tokenizer, 'eos_token', 'N/A')}")
            print(f"   Vocab size: {len(tokenizer):,}")
            print()

# === DEMO ===

print("🔤 Advanced Tokenization Comparison\n")
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

print("\n📌 Test Case 1: Simple Sentence\n")
comp.compare_on_text(test_texts[0])

print("\n📌 Test Case 2: Complex Words (OOV Handling)\n")
comp.compare_on_text(test_texts[2])

print("\n📌 Test Case 3: Contractions & Affixes\n")
comp.compare_on_text(test_texts[3])

# Special tokens
print("\n" + "="*70)
comp.special_tokens_analysis()

# Vocabulary coverage analysis
print("="*70)
print("\n📊 Vocabulary Coverage Analysis\n")

vocab_stats = comp.analyze_vocabulary_coverage(test_texts)

for name, token_counts in vocab_stats.items():
    avg_tokens = np.mean(token_counts)
    print(f"{name}:")
    print(f"   Average tokens per text: {avg_tokens:.2f}")
    print(f"   Range: {min(token_counts)}-{max(token_counts)}")
    print()

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Token count comparison across test texts
ax = axes[0, 0]
x = np.arange(len(test_texts))
width = 0.25

for i, (name, token_counts) in enumerate(vocab_stats.items()):
    ax.bar(x + i*width, token_counts, width, label=name, alpha=0.8)

ax.set_xlabel('Test Text')
ax.set_ylabel('Number of Tokens')
ax.set_title('Token Count Comparison')
ax.set_xticks(x + width)
ax.set_xticklabels([f'Text {i+1}' for i in range(len(test_texts))])
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)

# 2. Average efficiency (fewer tokens = more efficient)
ax = axes[0, 1]
avg_tokens = [np.mean(counts) for counts in vocab_stats.values()]
names = list(vocab_stats.keys())

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bars = ax.barh(names, avg_tokens, color=colors, alpha=0.7)
ax.set_xlabel('Average Tokens per Text')
ax.set_title('Tokenizer Efficiency (Lower = More Efficient)')
ax.grid(axis='x', alpha=0.3)

for bar, val in zip(bars, avg_tokens):
    ax.text(val + 0.2, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}', ha='left', va='center', fontweight='bold')

# 3. Vocabulary size comparison
ax = axes[1, 0]
vocab_sizes = [len(tokenizer) for tokenizer in comp.tokenizers.values()]
names_short = ['GPT-2\n(BPE)', 'BERT\n(WordPiece)', 'T5\n(SentencePiece)']

bars = ax.bar(names_short, vocab_sizes, color=colors, alpha=0.7)
ax.set_ylabel('Vocabulary Size')
ax.set_title('Tokenizer Vocabulary Size')
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

ax.set_xlabel('Token Rank')
ax.set_ylabel('Frequency (log scale)')
ax.set_title('Token Frequency Distribution')
ax.set_yscale('log')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('tokenizer_comparison.png', dpi=150, bbox_inches='tight')
print("\n📊 Gráfico salvo: tokenizer_comparison.png")

print("\n✅ Tokenizer comparison completo!")
print("\n💡 KEY INSIGHTS:")
print("   - BPE (GPT-2): Character-level fallback, good for rare words")
print("   - WordPiece (BERT): Subword units, balanced efficiency")
print("   - SentencePiece (T5): Language-agnostic, no pre-tokenization")
print("   - Trade-off: Vocab size vs. token sequence length")
print("\n💡 BEST PRACTICES:")
print("   - Choose tokenizer based on model architecture")
print("   - Consider vocab size for memory constraints")
print("   - Test OOV (out-of-vocabulary) handling")
print("   - Monitor token sequence length (affects inference cost)")
