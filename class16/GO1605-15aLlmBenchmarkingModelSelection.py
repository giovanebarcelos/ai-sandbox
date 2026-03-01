# GO1605-15aLlmBenchmarkingModelSelection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List
import seaborn as sns

class LLMBenchmark:
    """
    Sistema de benchmark para comparar LLMs

    Benchmarks:
    - MMLU (reasoning)
    - HumanEval (code generation)
    - GSM8K (math)
    - MT-Bench (chat quality)
    - TruthfulQA (factuality)
    """

    def __init__(self):
        # Benchmark scores (2024-2025 approximate)
        self.models = {
            'GPT-4o': {
                'params': '200B',
                'mmlu': 86.5,
                'humaneval': 88.0,
                'gsm8k': 92.0,
                'mt_bench': 9.1,
                'truthfulqa': 78.0,
                'cost_per_1m': 2.5,
                'latency_ms': 50,
                'open_source': False
            },
            'Claude Sonnet 3.5': {
                'params': '~200B',
                'mmlu': 88.7,
                'humaneval': 92.0,
                'gsm8k': 96.4,
                'mt_bench': 9.2,
                'truthfulqa': 85.0,
                'cost_per_1m': 3.0,
                'latency_ms': 45,
                'open_source': False
            },
            'Llama 3.1 70B': {
                'params': '70B',
                'mmlu': 82.0,
                'humaneval': 80.5,
                'gsm8k': 83.0,
                'mt_bench': 8.6,
                'truthfulqa': 72.0,
                'cost_per_1m': 0.9,
                'latency_ms': 70,
                'open_source': True
            },
            'Llama 3.1 8B': {
                'params': '8B',
                'mmlu': 69.0,
                'humaneval': 62.0,
                'gsm8k': 56.0,
                'mt_bench': 7.8,
                'truthfulqa': 58.0,
                'cost_per_1m': 0.2,
                'latency_ms': 25,
                'open_source': True
            },
            'Mistral 7B': {
                'params': '7B',
                'mmlu': 62.5,
                'humaneval': 40.2,
                'gsm8k': 52.2,
                'mt_bench': 7.3,
                'truthfulqa': 55.0,
                'cost_per_1m': 0.25,
                'latency_ms': 22,
                'open_source': True
            },
            'Phi-3-mini': {
                'params': '3.8B',
                'mmlu': 69.0,
                'humaneval': 58.0,
                'gsm8k': 82.5,
                'mt_bench': 7.5,
                'truthfulqa': 60.0,
                'cost_per_1m': 0.15,
                'latency_ms': 18,
                'open_source': True
            },
        }

    def compare_models(self, benchmark: str) -> pd.DataFrame:
        """Compare models on specific benchmark"""
        data = []
        for model_name, metrics in self.models.items():
            data.append({
                'Model': model_name,
                'Score': metrics[benchmark],
                'Open Source': 'Yes' if metrics['open_source'] else 'No'
            })

        df = pd.DataFrame(data).sort_values('Score', ascending=False)
        return df

    def calculate_overall_score(self) -> Dict:
        """Calculate weighted overall score"""
        weights = {
            'mmlu': 0.25,
            'humaneval': 0.20,
            'gsm8k': 0.20,
            'mt_bench': 0.20,
            'truthfulqa': 0.15
        }

        overall_scores = {}

        for model_name, metrics in self.models.items():
            score = sum(metrics[bench] * weight for bench, weight in weights.items())
            overall_scores[model_name] = score

        return overall_scores

    def recommend_model(self, use_case: str, budget: str) -> List[str]:
        """
        Recommend model based on use case and budget

        Use cases: 'general', 'code', 'math', 'chat', 'factual'
        Budget: 'high', 'medium', 'low'
        """
        recommendations = []

        if use_case == 'code':
            # Prioritize HumanEval
            sorted_models = sorted(self.models.items(), 
                                 key=lambda x: x[1]['humaneval'], 
                                 reverse=True)
        elif use_case == 'math':
            sorted_models = sorted(self.models.items(), 
                                 key=lambda x: x[1]['gsm8k'], 
                                 reverse=True)
        elif use_case == 'factual':
            sorted_models = sorted(self.models.items(), 
                                 key=lambda x: x[1]['truthfulqa'], 
                                 reverse=True)
        else:
            # General: use overall score
            overall_scores = self.calculate_overall_score()
            sorted_models = sorted(self.models.items(), 
                                 key=lambda x: overall_scores[x[0]], 
                                 reverse=True)

        # Filter by budget
        if budget == 'low':
            filtered = [(name, metrics) for name, metrics in sorted_models 
                       if metrics['cost_per_1m'] < 1.0]
        elif budget == 'medium':
            filtered = [(name, metrics) for name, metrics in sorted_models 
                       if metrics['cost_per_1m'] < 3.0]
        else:
            filtered = sorted_models

        return [name for name, _ in filtered[:3]]

# === DEMO ===

print("📊 LLM Benchmarking & Model Selection\n")
print("="*70)

benchmark = LLMBenchmark()

# Compare on MMLU (general reasoning)
print("\n📌 MMLU Benchmark (General Reasoning):\n")
mmlu_results = benchmark.compare_models('mmlu')
for _, row in mmlu_results.iterrows():
    print(f"   {row['Model']:<25} {row['Score']:.1f}  [{row['Open Source']}]")

# Compare on HumanEval (code)
print("\n📌 HumanEval Benchmark (Code Generation):\n")
humaneval_results = benchmark.compare_models('humaneval')
for _, row in humaneval_results.iterrows():
    print(f"   {row['Model']:<25} {row['Score']:.1f}%  [{row['Open Source']}]")

# Overall scores
print("\n📌 Overall Scores (Weighted Average):\n")
overall_scores = benchmark.calculate_overall_score()
for model, score in sorted(overall_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"   {model:<25} {score:.1f}")

# Recommendations
print("\n📌 Recommendations:\n")

use_cases = [
    ('general', 'low'),
    ('code', 'medium'),
    ('math', 'low'),
]

for use_case, budget in use_cases:
    recs = benchmark.recommend_model(use_case, budget)
    print(f"   {use_case.capitalize()} task, {budget} budget:")
    for i, model in enumerate(recs, 1):
        print(f"      {i}. {model}")
    print()

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Radar chart - model capabilities
ax = axes[0, 0]

models_to_plot = ['GPT-4o', 'Claude Sonnet 3.5', 'Llama 3.1 70B', 'Phi-3-mini']
benchmarks_radar = ['mmlu', 'humaneval', 'gsm8k', 'mt_bench', 'truthfulqa']

angles = np.linspace(0, 2 * np.pi, len(benchmarks_radar), endpoint=False).tolist()
angles += angles[:1]

ax = plt.subplot(221, projection='polar')

for model in models_to_plot:
    values = [benchmark.models[model][b] for b in benchmarks_radar]
    # Normalize to 0-100
    values = [v if b != 'mt_bench' else v * 10 for v, b in zip(values, benchmarks_radar)]
    values += values[:1]

    ax.plot(angles, values, 'o-', linewidth=2, label=model)
    ax.fill(angles, values, alpha=0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(['MMLU', 'HumanEval', 'GSM8K', 'MT-Bench', 'TruthfulQA'])
ax.set_ylim(0, 100)
ax.set_title('Model Capabilities (Radar)', y=1.08)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
ax.grid(True)

# 2. Cost vs Performance
ax = axes[0, 1]

for model_name, metrics in benchmark.models.items():
    overall = overall_scores[model_name]
    cost = metrics['cost_per_1m']
    color = 'green' if metrics['open_source'] else 'red'
    marker = 'o' if metrics['open_source'] else 's'

    ax.scatter(cost, overall, s=200, alpha=0.6, c=color, marker=marker)
    ax.annotate(model_name, (cost, overall), xytext=(5, 5), 
               textcoords='offset points', fontsize=8)

ax.set_xlabel('Cost per 1M tokens ($)')
ax.set_ylabel('Overall Score')
ax.set_title('Cost vs Performance')
ax.grid(alpha=0.3)
ax.legend(['Open Source', 'Proprietary'], loc='lower right')

# 3. Benchmark comparison
ax = axes[1, 0]

models_comp = ['GPT-4o', 'Llama 3.1 70B', 'Phi-3-mini']
benchmarks_comp = ['MMLU', 'HumanEval', 'GSM8K', 'MT-Bench', 'TruthfulQA']

x = np.arange(len(benchmarks_comp))
width = 0.25

for i, model in enumerate(models_comp):
    values = [benchmark.models[model][b.lower().replace('-', '_')] for b in benchmarks_comp]
    # Normalize MT-Bench to 0-100
    values[3] *= 10

    ax.bar(x + i*width, values, width, label=model, alpha=0.8)

ax.set_xlabel('Benchmark')
ax.set_ylabel('Score')
ax.set_title('Performance Across Benchmarks')
ax.set_xticks(x + width)
ax.set_xticklabels(benchmarks_comp, rotation=45, ha='right')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)

# 4. Latency comparison
ax = axes[1, 1]

models_lat = list(benchmark.models.keys())
latencies = [benchmark.models[m]['latency_ms'] for m in models_lat]
colors_lat = ['green' if benchmark.models[m]['open_source'] else 'red' 
             for m in models_lat]

bars = ax.barh(models_lat, latencies, color=colors_lat, alpha=0.7)
ax.set_xlabel('Latency (ms)')
ax.set_title('Inference Latency Comparison')
ax.grid(axis='x', alpha=0.3)

for bar, lat in zip(bars, latencies):
    width = bar.get_width()
    ax.text(width + 2, bar.get_y() + bar.get_height()/2,
            f'{lat}ms', ha='left', va='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('llm_benchmarking.png', dpi=150, bbox_inches='tight')
print("\n📊 Gráfico salvo: llm_benchmarking.png")

print("\n✅ LLM benchmarking implementado!")
print("\n💡 BENCHMARK GUIDE:")
print("   MMLU: General knowledge & reasoning (57 subjects)")
print("   HumanEval: Code generation (164 programming problems)")
print("   GSM8K: Math word problems (grade school)")
print("   MT-Bench: Multi-turn conversation quality")
print("   TruthfulQA: Factual accuracy & avoiding misinformation")
print("\n💡 MODEL SELECTION TIPS:")
print("   - Claude Sonnet: Best overall, excellent for complex tasks")
print("   - GPT-4o: Strong general purpose, fast")
print("   - Llama 3.1 70B: Best open-source, self-host option")
print("   - Phi-3-mini: Surprising quality for size, low cost")
print("   - Consider task-specific strengths!")
