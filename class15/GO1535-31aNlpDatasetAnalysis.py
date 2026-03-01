# GO1535-31aNlpDatasetAnalysis
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class NLPDatasetAnalyzer:
    """
    NLP Dataset Analysis Tool

    Provides:
    - Dataset statistics (size, vocab, avg length)
    - Task type classification
    - Benchmark scores
    - Dataset recommendations
    """

    def __init__(self):
        # Popular NLP datasets with metadata
        self.datasets = {
            'IMDB Reviews': {
                'task': 'Sentiment Analysis',
                'samples': 50000,
                'classes': 2,
                'avg_words': 234,
                'language': 'English',
                'difficulty': 'Easy',
                'benchmark_acc': 0.943,
                'domain': 'Movie reviews'
            },
            '20 Newsgroups': {
                'task': 'Text Classification',
                'samples': 18846,
                'classes': 20,
                'avg_words': 150,
                'language': 'English',
                'difficulty': 'Medium',
                'benchmark_acc': 0.851,
                'domain': 'News articles'
            },
            'SQuAD 2.0': {
                'task': 'Question Answering',
                'samples': 150000,
                'classes': None,
                'avg_words': 120,
                'language': 'English',
                'difficulty': 'Hard',
                'benchmark_acc': 0.892,  # F1 score
                'domain': 'Wikipedia'
            },
            'CoNLL-2003': {
                'task': 'Named Entity Recognition',
                'samples': 20000,
                'classes': 4,  # PER, LOC, ORG, MISC
                'avg_words': 25,
                'language': 'English',
                'difficulty': 'Hard',
                'benchmark_acc': 0.934,  # F1 score
                'domain': 'News'
            },
            'AG News': {
                'task': 'News Classification',
                'samples': 120000,
                'classes': 4,
                'avg_words': 45,
                'language': 'English',
                'difficulty': 'Easy',
                'benchmark_acc': 0.946,
                'domain': 'News headlines'
            },
            'SST-2': {
                'task': 'Sentiment Analysis',
                'samples': 67349,
                'classes': 2,
                'avg_words': 19,
                'language': 'English',
                'difficulty': 'Medium',
                'benchmark_acc': 0.972,  # BERT
                'domain': 'Movie reviews (fine-grained)'
            },
            'WikiText-103': {
                'task': 'Language Modeling',
                'samples': 28475,
                'classes': None,
                'avg_words': 3600,
                'language': 'English',
                'difficulty': 'Medium',
                'benchmark_acc': 0.184,  # Perplexity (lower better)
                'domain': 'Wikipedia articles'
            },
            'CNN/Daily Mail': {
                'task': 'Summarization',
                'samples': 311971,
                'classes': None,
                'avg_words': 766,
                'language': 'English',
                'difficulty': 'Hard',
                'benchmark_acc': 0.423,  # ROUGE-L
                'domain': 'News articles'
            }
        }

    def get_summary(self):
        """Get dataset summary statistics"""
        df = pd.DataFrame(self.datasets).T
        return df

    def recommend_dataset(self, task_type):
        """Recommend datasets for specific task"""
        recommendations = []

        for name, meta in self.datasets.items():
            if task_type.lower() in meta['task'].lower():
                recommendations.append({
                    'name': name,
                    'samples': meta['samples'],
                    'difficulty': meta['difficulty'],
                    'benchmark': meta['benchmark_acc']
                })

        return sorted(recommendations, key=lambda x: x['samples'], reverse=True)

    def compare_datasets(self, dataset_names):
        """Compare multiple datasets"""
        comparison = {}

        for name in dataset_names:
            if name in self.datasets:
                comparison[name] = self.datasets[name]

        return pd.DataFrame(comparison).T

# === DEMO ===

print("📊 NLP Dataset Analysis Demo\n")
print("="*70)

# Initialize analyzer
analyzer = NLPDatasetAnalyzer()

# Get all datasets
print("📊 DATASET OVERVIEW\n")
df = analyzer.get_summary()
print(df[['task', 'samples', 'classes', 'difficulty']].to_string())

print("\n" + "="*70)
print("\n🎯 RECOMMENDATIONS BY TASK\n")

# Recommendations
tasks = ['Sentiment Analysis', 'Classification', 'Question Answering']

for task in tasks:
    recommendations = analyzer.recommend_dataset(task)
    print(f"{task}:")
    for rec in recommendations:
        print(f"   - {rec['name']}: {rec['samples']:,} samples ({rec['difficulty']})")
    print()

print("="*70)
print("\n🔍 DETAILED COMPARISON: Sentiment Analysis\n")

sentiment_datasets = ['IMDB Reviews', 'SST-2']
comparison = analyzer.compare_datasets(sentiment_datasets)
print(comparison[['samples', 'avg_words', 'difficulty', 'benchmark_acc']].to_string())

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Dataset sizes
ax = axes[0, 0]
dataset_names = list(analyzer.datasets.keys())
sizes = [analyzer.datasets[name]['samples'] for name in dataset_names]

colors_map = {'Easy': 'lightgreen', 'Medium': 'gold', 'Hard': 'lightcoral'}
colors = [colors_map[analyzer.datasets[name]['difficulty']] for name in dataset_names]

ax.barh(dataset_names, sizes, color=colors, alpha=0.7)
ax.set_xlabel('Number of Samples')
ax.set_title('Dataset Sizes')
ax.set_xscale('log')

for i, size in enumerate(sizes):
    ax.text(size * 1.1, i, f"{size:,}", va='center', fontsize=8)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, alpha=0.7, label=l) for l, c in colors_map.items()]
ax.legend(handles=legend_elements, loc='lower right', title='Difficulty')

# 2. Average text length
ax = axes[0, 1]
avg_lengths = [analyzer.datasets[name]['avg_words'] for name in dataset_names]

ax.bar(range(len(dataset_names)), avg_lengths, color='skyblue', alpha=0.7)
ax.set_xticks(range(len(dataset_names)))
ax.set_xticklabels([name.split()[0] for name in dataset_names], rotation=45, ha='right')
ax.set_ylabel('Average Words per Sample')
ax.set_title('Text Length Distribution')
ax.grid(axis='y', alpha=0.3)

for i, length in enumerate(avg_lengths):
    ax.text(i, length + 50, str(length), ha='center', va='bottom', fontsize=9)

# 3. Task type distribution
ax = axes[1, 0]
task_counts = {}
for name in dataset_names:
    task = analyzer.datasets[name]['task']
    task_counts[task] = task_counts.get(task, 0) + 1

tasks_plot = list(task_counts.keys())
counts_plot = list(task_counts.values())

ax.pie(counts_plot, labels=tasks_plot, autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
ax.set_title('Dataset Distribution by Task')

# 4. Benchmark performance
ax = axes[1, 1]
benchmark_datasets = ['IMDB Reviews', '20 Newsgroups', 'AG News', 'SST-2', 'CoNLL-2003']
benchmarks = [analyzer.datasets[name]['benchmark_acc'] for name in benchmark_datasets]

colors_bench = [colors_map[analyzer.datasets[name]['difficulty']] for name in benchmark_datasets]

bars = ax.bar(range(len(benchmark_datasets)), benchmarks, color=colors_bench, alpha=0.7)
ax.set_xticks(range(len(benchmark_datasets)))
ax.set_xticklabels([name.split()[0] for name in benchmark_datasets], rotation=45, ha='right')
ax.set_ylabel('Benchmark Score')
ax.set_title('State-of-the-Art Performance')
ax.set_ylim(0, 1.1)
ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90% threshold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

for bar, score in zip(bars, benchmarks):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('nlp_dataset_analysis.png', dpi=150, bbox_inches='tight')
print("📊 Gráfico salvo: nlp_dataset_analysis.png")

print("\n✅ Dataset analysis completo!")
print("\n💡 CHOOSING A DATASET:")
print("   - Beginner: IMDB, AG News (clear labels, large size)")
print("   - Intermediate: 20 Newsgroups, SST-2 (more nuanced)")
print("   - Advanced: SQuAD, CoNLL-2003 (complex tasks)")
print("\n💡 DATASET SOURCES:")
print("   - Hugging Face Datasets: datasets.load_dataset()")
print("   - TensorFlow Datasets: tfds.load()")
print("   - PyTorch torchtext: torchtext.datasets")
print("   - Kaggle: kaggle.com/datasets")
print("\n💡 CONSIDERATIONS:")
print("   - Size: Larger = better models, longer training")
print("   - Domain: Match your application domain")
print("   - Language: Most datasets are English")
print("   - Balance: Check class distribution")
