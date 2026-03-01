# GO1626-34aQuickModelEvaluationMetrics
from transformers import pipeline
import numpy as np
from typing import List, Dict

def evaluate_llm_generation(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Evaluate LLM generation quality

    Metrics:
    - Exact Match (EM): Percentage of exact matches
    - Average Length: Token count comparison
    - Fluency Score: Grammar/coherence (simplified)
    """

    # Exact Match
    exact_matches = sum(p.strip().lower() == r.strip().lower() 
                       for p, r in zip(predictions, references))
    em_score = exact_matches / len(predictions) * 100

    # Length comparison
    pred_lengths = [len(p.split()) for p in predictions]
    ref_lengths = [len(r.split()) for r in references]
    avg_pred_len = np.mean(pred_lengths)
    avg_ref_len = np.mean(ref_lengths)

    # Simple fluency: check punctuation and capitalization
    fluency_scores = []
    for pred in predictions:
        score = 0
        if pred[0].isupper():  # Starts with capital
            score += 0.5
        if pred[-1] in '.!?':  # Ends with punctuation
            score += 0.5
        fluency_scores.append(score)
    avg_fluency = np.mean(fluency_scores) * 100

    return {
        'exact_match': em_score,
        'avg_prediction_length': avg_pred_len,
        'avg_reference_length': avg_ref_len,
        'fluency_score': avg_fluency
    }

# Demo
predictions = [
    "Machine learning is a subset of AI.",
    "Python is a programming language",
    "Transformers use attention mechanisms."
]

references = [
    "Machine learning is a subset of AI.",
    "Python is a popular programming language.",
    "Transformers use self-attention mechanisms."
]

print("📊 LLM Evaluation Metrics\n")
print("="*60)
metrics = evaluate_llm_generation(predictions, references)
for metric, value in metrics.items():
    print(f"   {metric}: {value:.2f}")

print("\n💡 Key Evaluation Frameworks:")
print("   - BLEU: N-gram overlap (machine translation)")
print("   - ROUGE: Recall-oriented (summarization)")
print("   - BERTScore: Semantic similarity")
print("   - Perplexity: Language model quality")
print("   - Human evaluation: Gold standard for quality")
