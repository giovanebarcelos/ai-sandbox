# GO1718-30bRagEvaluationMetrics
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class RAGEvaluation:
    """Resultado de avaliação RAG"""
    # Retrieval metrics
    retrieval_precision: float
    retrieval_recall: float
    retrieval_mrr: float  # Mean Reciprocal Rank

    # Generation metrics
    answer_relevance: float
    answer_faithfulness: float  # Fidelidade aos docs
    answer_completeness: float

    # Overall
    ragas_score: float  # RAGAS = Retrieval Augmented Generation Assessment

class RAGEvaluator:
    """
    Avalia sistemas RAG em múltiplas dimensões:

    1. RETRIEVAL QUALITY
       - Precision@K: % docs relevantes recuperados
       - Recall@K: % docs relevantes encontrados
       - MRR: posição do primeiro doc relevante
       - NDCG: ranking quality

    2. GENERATION QUALITY
       - Relevance: resposta responde a pergunta?
       - Faithfulness: resposta é fiel aos docs?
       - Completeness: resposta é completa?
       - Conciseness: resposta é concisa?

    3. END-TO-END
       - RAGAS score: métrica composta
       - Latency: tempo de resposta
       - Cost: tokens/custo
    """

    def __init__(self):
        self.results = []

    def evaluate_retrieval(self, 
                          retrieved_docs: List[str],
                          relevant_docs: List[str],
                          k: int = 5) -> Dict[str, float]:
        """Avalia qualidade da recuperação"""
        retrieved_set = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)

        # Precision@K
        if len(retrieved_set) == 0:
            precision = 0.0
        else:
            precision = len(retrieved_set & relevant_set) / len(retrieved_set)

        # Recall@K
        if len(relevant_set) == 0:
            recall = 1.0
        else:
            recall = len(retrieved_set & relevant_set) / len(relevant_set)

        # Mean Reciprocal Rank (MRR)
        mrr = 0.0
        for i, doc in enumerate(retrieved_docs, 1):
            if doc in relevant_set:
                mrr = 1.0 / i
                break

        # F1 score
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return {
            'precision@k': precision,
            'recall@k': recall,
            'f1@k': f1,
            'mrr': mrr
        }

    def evaluate_generation(self,
                           question: str,
                           answer: str,
                           retrieved_docs: List[str],
                           ground_truth: str = None) -> Dict[str, float]:
        """Avalia qualidade da geração"""

        # 1. Relevance (simulated - in real use LLM-as-judge)
        # Check if answer contains question keywords
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        relevance = len(q_words & a_words) / len(q_words) if q_words else 0.0

        # 2. Faithfulness (answer based on docs?)
        # Check if answer tokens appear in docs
        doc_text = ' '.join(retrieved_docs).lower()
        answer_tokens = answer.lower().split()
        faithful_tokens = sum(1 for token in answer_tokens if token in doc_text)
        faithfulness = faithful_tokens / len(answer_tokens) if answer_tokens else 0.0

        # 3. Completeness (answer length relative to ground truth)
        if ground_truth:
            completeness = min(len(answer) / len(ground_truth), 1.0)
        else:
            # Heuristic: answers should be 50-500 chars
            if len(answer) < 50:
                completeness = len(answer) / 50
            elif len(answer) > 500:
                completeness = 500 / len(answer)
            else:
                completeness = 1.0

        # 4. Conciseness (not too verbose)
        # Penalize if answer is more than 2x ground truth
        if ground_truth:
            conciseness = min(len(ground_truth) / len(answer), 1.0)
        else:
            conciseness = min(200 / len(answer), 1.0) if answer else 0.0

        return {
            'relevance': relevance,
            'faithfulness': faithfulness,
            'completeness': completeness,
            'conciseness': conciseness
        }

    def compute_ragas_score(self, 
                           retrieval_metrics: Dict,
                           generation_metrics: Dict) -> float:
        """
        RAGAS Score (Retrieval Augmented Generation Assessment)

        Combina retrieval e generation metrics
        """
        # Weighted average of key metrics
        ragas = (
            0.2 * retrieval_metrics['precision@k'] +
            0.2 * retrieval_metrics['recall@k'] +
            0.2 * generation_metrics['relevance'] +
            0.3 * generation_metrics['faithfulness'] +
            0.1 * generation_metrics['completeness']
        )

        return ragas

    def evaluate(self,
                 question: str,
                 answer: str,
                 retrieved_docs: List[str],
                 relevant_docs: List[str],
                 ground_truth: str = None) -> RAGEvaluation:
        """Avaliação completa"""

        # Evaluate retrieval
        ret_metrics = self.evaluate_retrieval(retrieved_docs, relevant_docs)

        # Evaluate generation
        gen_metrics = self.evaluate_generation(
            question, answer, retrieved_docs, ground_truth
        )

        # Compute RAGAS
        ragas = self.compute_ragas_score(ret_metrics, gen_metrics)

        result = RAGEvaluation(
            retrieval_precision=ret_metrics['precision@k'],
            retrieval_recall=ret_metrics['recall@k'],
            retrieval_mrr=ret_metrics['mrr'],
            answer_relevance=gen_metrics['relevance'],
            answer_faithfulness=gen_metrics['faithfulness'],
            answer_completeness=gen_metrics['completeness'],
            ragas_score=ragas
        )

        self.results.append({
            'question': question,
            'evaluation': result
        })

        return result

# === DEMO ===

evaluator = RAGEvaluator()

# Test cases
test_cases = [
    {
        'question': 'What is machine learning?',
        'answer': 'Machine learning is a subset of AI that enables computers to learn from data without explicit programming.',
        'retrieved_docs': [
            'Machine learning is a subset of artificial intelligence.',
            'Deep learning uses neural networks with multiple layers.',
            'Supervised learning requires labeled training data.'
        ],
        'relevant_docs': [
            'Machine learning is a subset of artificial intelligence.',
            'Machine learning enables computers to learn from data.'
        ],
        'ground_truth': 'Machine learning is a branch of AI focused on learning from data.'
    },
    {
        'question': 'How do neural networks work?',
        'answer': 'Neural networks consist of interconnected nodes that process information in layers.',
        'retrieved_docs': [
            'Neural networks are inspired by biological neurons.',
            'Backpropagation trains neural networks by adjusting weights.',
            'Activation functions introduce non-linearity.'
        ],
        'relevant_docs': [
            'Neural networks are inspired by biological neurons.',
            'Backpropagation trains neural networks by adjusting weights.'
        ],
        'ground_truth': 'Neural networks use layers of neurons with weighted connections trained via backpropagation.'
    },
    {
        'question': 'Explain transformers',
        'answer': 'Transformers use attention.',  # Poor answer
        'retrieved_docs': [
            'Transformers revolutionized NLP with self-attention mechanisms.',
            'BERT and GPT are based on transformer architecture.'
        ],
        'relevant_docs': [
            'Transformers revolutionized NLP with self-attention mechanisms.'
        ],
        'ground_truth': 'Transformers are neural network architectures using self-attention to process sequences in parallel.'
    }
]

print("📊 Avaliando Sistema RAG\n")
print("="*70)

for i, tc in enumerate(test_cases, 1):
    print(f"\n📌 Test Case {i}")
    print(f"Question: {tc['question']}")
    print(f"Answer: {tc['answer']}\n")

    result = evaluator.evaluate(
        tc['question'],
        tc['answer'],
        tc['retrieved_docs'],
        tc['relevant_docs'],
        tc['ground_truth']
    )

    print(f"RETRIEVAL METRICS:")
    print(f"  Precision@K: {result.retrieval_precision:.3f}")
    print(f"  Recall@K: {result.retrieval_recall:.3f}")
    print(f"  MRR: {result.retrieval_mrr:.3f}")

    print(f"\nGENERATION METRICS:")
    print(f"  Relevance: {result.answer_relevance:.3f}")
    print(f"  Faithfulness: {result.answer_faithfulness:.3f}")
    print(f"  Completeness: {result.answer_completeness:.3f}")

    print(f"\n⭐ RAGAS SCORE: {result.ragas_score:.3f}")

# Visualize results
results = [r['evaluation'] for r in evaluator.results]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Radar chart for each test case
ax = axes[0, 0]
categories = ['Precision', 'Recall', 'MRR', 'Relevance', 'Faithfulness', 'Completeness']
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

for i, res in enumerate(results, 1):
    values = [
        res.retrieval_precision,
        res.retrieval_recall,
        res.retrieval_mrr,
        res.answer_relevance,
        res.answer_faithfulness,
        res.answer_completeness
    ]
    values += values[:1]

    ax.plot(angles, values, 'o-', linewidth=2, label=f'Test {i}')
    ax.fill(angles, values, alpha=0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=8)
ax.set_ylim(0, 1)
ax.set_title('RAG Evaluation: All Metrics')
ax.legend(loc='upper right')
ax.grid(True)

# 2. RAGAS scores comparison
ax = axes[0, 1]
ragas_scores = [r.ragas_score for r in results]
colors = ['lightgreen' if s >= 0.7 else 'yellow' if s >= 0.5 else 'lightcoral' for s in ragas_scores]
bars = ax.barh([f'Test {i}' for i in range(1, len(results)+1)], ragas_scores, color=colors, alpha=0.7)
ax.set_xlabel('RAGAS Score')
ax.set_title('Overall RAGAS Scores')
ax.axvline(0.7, color='green', linestyle='--', alpha=0.5, label='Good threshold')
ax.axvline(0.5, color='orange', linestyle='--', alpha=0.5, label='Acceptable threshold')
ax.legend()
ax.grid(axis='x', alpha=0.3)

for bar, score in zip(bars, ragas_scores):
    width = bar.get_width()
    ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
            f'{score:.3f}', ha='left', va='center')

# 3. Retrieval vs Generation
ax = axes[1, 0]
retrieval_avg = np.mean([
    [r.retrieval_precision, r.retrieval_recall, r.retrieval_mrr] 
    for r in results
], axis=1)
generation_avg = np.mean([
    [r.answer_relevance, r.answer_faithfulness, r.answer_completeness]
    for r in results
], axis=1)

x = np.arange(len(results))
width = 0.35

ax.bar(x - width/2, retrieval_avg, width, label='Retrieval', color='skyblue', alpha=0.8)
ax.bar(x + width/2, generation_avg, width, label='Generation', color='lightcoral', alpha=0.8)

ax.set_ylabel('Average Score')
ax.set_title('Retrieval vs Generation Quality')
ax.set_xticks(x)
ax.set_xticklabels([f'Test {i}' for i in range(1, len(results)+1)])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 4. Metric correlation heatmap
ax = axes[1, 1]
import pandas as pd
metric_data = {
    'Precision': [r.retrieval_precision for r in results],
    'Recall': [r.retrieval_recall for r in results],
    'Relevance': [r.answer_relevance for r in results],
    'Faithfulness': [r.answer_faithfulness for r in results],
    'RAGAS': [r.ragas_score for r in results]
}
df = pd.DataFrame(metric_data)
corr = df.corr()

import seaborn as sns
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title('Metric Correlations')

plt.tight_layout()
plt.savefig('rag_evaluation_results.png', dpi=150, bbox_inches='tight')
print("\n\n📊 Gráfico salvo: rag_evaluation_results.png")

# Summary
avg_ragas = np.mean([r.ragas_score for r in results])
print(f"\n📈 SUMMARY:")
print(f"   Average RAGAS Score: {avg_ragas:.3f}")
print(f"   Tests passed (>0.7): {sum(1 for r in results if r.ragas_score >= 0.7)}/{len(results)}")
print(f"\n💡 RECOMMENDATIONS:")
if avg_ragas < 0.5:
    print("   ⚠️  Sistema precisa de melhorias significativas")
elif avg_ragas < 0.7:
    print("   ⚡ Sistema funcional, mas pode melhorar")
else:
    print("   ✅ Sistema com boa qualidade")

print("\n✅ Framework de avaliação RAG implementado!")
