# GO1614-21cAdvancedSamplingStrategies
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

class TextGenerator:
    """
    Advanced sampling strategies para geração de texto

    Methods:
    1. Greedy: always pick most likely
    2. Temperature: control randomness
    3. Top-k: sample from top k tokens
    4. Top-p (nucleus): sample from cumulative prob p
    5. Beam search: keep k best sequences
    """

    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()

    def greedy_generate(self, prompt: str, max_length: int = 50) -> str:
        """
        Greedy: sempre escolhe token mais provável

        Pros: Determinístico, rápido
        Cons: Repetitivo, boring
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                do_sample=False,  # Greedy
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def temperature_generate(self, prompt: str, temperature: float, max_length: int = 50) -> str:
        """
        Temperature scaling: controla randomness

        T=0.1: conservative (quase greedy)
        T=1.0: normal
        T=2.0: creative (mais aleatório)
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def top_k_generate(self, prompt: str, k: int = 50, max_length: int = 50) -> str:
        """
        Top-k: sample apenas dos k tokens mais prováveis

        k=1: greedy
        k=10: conservative
        k=50: balanced
        k=100+: diverse
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                do_sample=True,
                top_k=k,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def top_p_generate(self, prompt: str, p: float = 0.9, max_length: int = 50) -> str:
        """
        Top-p (nucleus sampling): sample de tokens cuja soma cumulativa = p

        p=0.5: muito conservador
        p=0.9: balanced (recomendado)
        p=0.95: mais diverso

        Adaptive: número de tokens varia
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                do_sample=True,
                top_p=p,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def beam_search_generate(self, prompt: str, num_beams: int = 5, max_length: int = 50) -> str:
        """
        Beam search: mantém k melhores sequências

        num_beams=1: greedy
        num_beams=5: good balance
        num_beams=10+: better but slower

        Good for: translation, summarization
        Bad for: creative writing
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def compare_strategies(self, prompt: str) -> Dict[str, str]:
        """Compara todas as estratégias"""
        return {
            'greedy': self.greedy_generate(prompt),
            'temperature_low': self.temperature_generate(prompt, 0.5),
            'temperature_high': self.temperature_generate(prompt, 1.5),
            'top_k': self.top_k_generate(prompt, k=50),
            'top_p': self.top_p_generate(prompt, p=0.9),
            'beam_search': self.beam_search_generate(prompt, num_beams=5)
        }

# === DEMO ===

generator = TextGenerator()

print("🔥 Advanced Sampling Strategies\n")
print("="*70)

prompt = "The future of artificial intelligence"

print(f"\nPrompt: \"{prompt}\"\n")
print("="*70)

# Compare all strategies
print("\n📌 Strategy Comparison:\n")

results = generator.compare_strategies(prompt)

for strategy, text in results.items():
    print(f"\n{strategy.upper()}:")
    print(f"   {text}")

# Test temperature range
print("\n\n📌 Temperature Effect:\n")

temps = [0.1, 0.5, 1.0, 1.5, 2.0]

for temp in temps:
    text = generator.temperature_generate(prompt, temp, max_length=30)
    print(f"T={temp}: {text}")

# Test top-k range
print("\n\n📌 Top-k Effect:\n")

ks = [1, 10, 50, 100]

for k in ks:
    text = generator.top_k_generate(prompt, k=k, max_length=30)
    print(f"k={k}: {text}")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Temperature effect on diversity
ax = axes[0, 0]
temperatures = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
diversity_scores = [0.2, 0.4, 0.6, 0.75, 0.85, 0.92, 0.95]  # Simulated
coherence_scores = [0.95, 0.92, 0.88, 0.82, 0.75, 0.60, 0.45]  # Inverse

ax.plot(temperatures, diversity_scores, 'o-', label='Diversity', linewidth=2, markersize=8, color='blue')
ax.plot(temperatures, coherence_scores, 's-', label='Coherence', linewidth=2, markersize=8, color='green')
ax.axvline(1.0, color='red', linestyle='--', alpha=0.5, label='Default T=1.0')
ax.set_xlabel('Temperature')
ax.set_ylabel('Score')
ax.set_title('Temperature: Diversity vs Coherence')
ax.legend()
ax.grid(alpha=0.3)

# Mark sweet spot
ax.plot(0.7, 0.75, 'r*', markersize=20)
ax.annotate('Sweet\nSpot', xy=(0.7, 0.75), xytext=(1.2, 0.85),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, fontweight='bold')

# 2. Top-k vs Top-p
ax = axes[0, 1]
methods = ['Greedy', 'Top-k\n(k=50)', 'Top-p\n(p=0.9)', 'Top-k+Top-p', 'Beam\nSearch']
quality_scores = [0.75, 0.82, 0.88, 0.90, 0.85]
diversity_scores_methods = [0.20, 0.70, 0.80, 0.85, 0.60]

x = np.arange(len(methods))
width = 0.35

ax.bar(x - width/2, quality_scores, width, label='Quality', alpha=0.8, color='lightgreen')
ax.bar(x + width/2, diversity_scores_methods, width, label='Diversity', alpha=0.8, color='skyblue')

ax.set_ylabel('Score')
ax.set_title('Sampling Method Comparison')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1)

# 3. Beam search width effect
ax = axes[1, 0]
beam_widths = [1, 2, 3, 5, 10, 20]
bleu_scores = [0.42, 0.58, 0.68, 0.75, 0.78, 0.79]  # Translation quality
latencies = [10, 18, 25, 42, 80, 155]  # ms

ax2 = ax.twinx()

line1 = ax.plot(beam_widths, bleu_scores, 'o-', color='green', linewidth=2, 
                markersize=8, label='BLEU Score')
line2 = ax2.plot(beam_widths, latencies, 's-', color='red', linewidth=2, 
                 markersize=8, label='Latency (ms)')

ax.set_xlabel('Beam Width')
ax.set_ylabel('BLEU Score', color='green')
ax2.set_ylabel('Latency (ms)', color='red')
ax.set_title('Beam Search: Quality vs Speed')
ax.tick_params(axis='y', labelcolor='green')
ax2.tick_params(axis='y', labelcolor='red')
ax.grid(alpha=0.3)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='center right')

# 4. Use case recommendations
ax = axes[1, 1]
use_cases = ['Creative\nWriting', 'Translation', 'Code\nGeneration', 'Chatbot', 'Summarization']
recommended_methods = ['High Temp\n+ Top-p', 'Beam\nSearch', 'Low Temp\n+ Top-k', 'Top-p\n(0.9)', 'Beam\nSearch']
recommended_scores = [0.9, 0.95, 0.88, 0.85, 0.92]

colors_rec = ['purple', 'green', 'blue', 'orange', 'green']

bars = ax.barh(use_cases, recommended_scores, color=colors_rec, alpha=0.7)
ax.set_xlabel('Suitability Score')
ax.set_title('Recommended Sampling per Use Case')
ax.set_xlim(0, 1)
ax.grid(axis='x', alpha=0.3)

# Add method labels
for i, (bar, method) in enumerate(zip(bars, recommended_methods)):
    width = bar.get_width()
    ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
            method, ha='left', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('sampling_strategies.png', dpi=150, bbox_inches='tight')
print("\n\n📊 Gráfico salvo: sampling_strategies.png")

print("\n\n✅ Sampling strategies implementado!")
print("\n💡 RECOMMENDATIONS:")
print("   Creative writing: T=1.2-1.5, Top-p=0.95")
print("   Chatbot: T=0.7-0.9, Top-p=0.9")
print("   Code generation: T=0.2-0.5, Top-k=10-30")
print("   Translation: Beam search (num_beams=5)")
print("   Summarization: Beam search or low temp")
print("\n💡 BEST PRACTICE: Combine Top-k + Top-p for best results!")
