# GO1531-25aQuestionAnsweringSystem
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import matplotlib.pyplot as plt
import numpy as np

class QuestionAnsweringSystem:
    """
    Extractive Question Answering System

    Process:
    1. Tokenize question + context
    2. Pass through BERT-based QA model
    3. Extract answer span (start, end positions)
    4. Decode answer from tokens
    5. Compute confidence score
    """

    def __init__(self, model_name='deepset/roberta-base-squad2'):
        print(f"🤖 Loading QA model: {model_name}...\n")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)

        print("✅ Model loaded successfully!\n")

    def answer_question(self, question, context, return_details=False):
        """
        Answer a question based on context

        Returns:
        - answer: extracted text span
        - confidence: probability score
        - start_idx, end_idx: character positions in context
        """
        # Tokenize
        inputs = self.tokenizer(
            question,
            context,
            return_tensors='pt',
            truncation=True,
            max_length=512
        )

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get start and end logits
        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        # Get most likely start and end positions
        start_idx = torch.argmax(start_logits)
        end_idx = torch.argmax(end_logits)

        # Compute confidence (softmax of logits)
        start_probs = torch.softmax(start_logits, dim=0)
        end_probs = torch.softmax(end_logits, dim=0)

        confidence = (start_probs[start_idx] * end_probs[end_idx]).item()

        # Extract answer
        answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)

        # Find character positions in original context
        char_start = context.find(answer) if answer else -1
        char_end = char_start + len(answer) if char_start != -1 else -1

        result = {
            'question': question,
            'answer': answer,
            'confidence': confidence,
            'char_start': char_start,
            'char_end': char_end
        }

        if return_details:
            result['start_logits'] = start_logits.tolist()[:20]  # First 20 for viz
            result['end_logits'] = end_logits.tolist()[:20]

        return result

    def batch_qa(self, questions, context):
        """Answer multiple questions on same context"""
        results = []

        for question in questions:
            result = self.answer_question(question, context)
            results.append(result)

        return results

# === DEMO ===

print("❓ Question Answering System Demo\n")
print("="*70)

# Context paragraph
context = """
Python is a high-level, interpreted programming language created by Guido van Rossum 
and first released in 1991. It emphasizes code readability with significant indentation. 
Python is dynamically typed and garbage-collected. It supports multiple programming 
paradigms, including structured, object-oriented and functional programming. Python is 
often described as a "batteries included" language due to its comprehensive standard 
library. The language is widely used for web development, data science, machine learning, 
and automation scripts.
"""

print("📝 Context Paragraph:\n")
print(context)
print("\n" + "="*70)

# Initialize QA system
qa_system = QuestionAnsweringSystem()

# Questions
questions = [
    "Who created Python?",
    "When was Python first released?",
    "What programming paradigms does Python support?",
    "What is Python commonly used for?",
    "What is Python's type system?"
]

print("\n❓ QUESTION-ANSWER PAIRS:\n")

results = qa_system.batch_qa(questions, context)

for i, result in enumerate(results, 1):
    print(f"Q{i}: {result['question']}")
    print(f"A{i}: {result['answer']}")
    print(f"   Confidence: {result['confidence']:.4f}")
    print(f"   Position: [{result['char_start']}:{result['char_end']}]")
    print()

# Detailed analysis for one question
print("="*70)
print("\n🔍 DETAILED ANALYSIS: Question 1\n")

detailed = qa_system.answer_question(
    questions[0],
    context,
    return_details=True
)

print(f"Question: {detailed['question']}")
print(f"Answer: {detailed['answer']}")
print(f"Confidence: {detailed['confidence']:.4f}")
print(f"\nAnswer in context:")
print(f"...{context[max(0, detailed['char_start']-30):detailed['char_end']+30]}...")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Confidence scores
ax = axes[0, 0]
confidences = [r['confidence'] for r in results]
question_labels = [f"Q{i+1}" for i in range(len(questions))]

bars = ax.bar(question_labels, confidences, color='skyblue', alpha=0.7)
ax.set_ylabel('Confidence Score')
ax.set_title('Answer Confidence by Question')
ax.set_ylim(0, 1)
ax.axhline(y=0.5, color='r', linestyle='--', label='Threshold', alpha=0.5)
ax.legend()
ax.grid(axis='y', alpha=0.3)

for bar, conf in zip(bars, confidences):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{conf:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 2. Answer lengths
ax = axes[0, 1]
answer_lengths = [len(r['answer'].split()) for r in results]

ax.bar(question_labels, answer_lengths, color='lightgreen', alpha=0.7)
ax.set_ylabel('Answer Length (words)')
ax.set_title('Answer Length Distribution')
ax.grid(axis='y', alpha=0.3)

for i, length in enumerate(answer_lengths):
    ax.text(i, length + 0.2, str(length), ha='center', va='bottom', fontweight='bold')

# 3. Start logits (first question)
ax = axes[1, 0]
if 'start_logits' in detailed:
    start_logits = detailed['start_logits']
    ax.plot(start_logits, 'o-', color='blue', linewidth=2, markersize=5)
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Logit Score')
    ax.set_title('Start Position Logits (Q1)')
    ax.grid(alpha=0.3)
    ax.axvline(x=np.argmax(start_logits), color='red', linestyle='--', 
              label='Selected Start', alpha=0.7)
    ax.legend()

# 4. Answer position visualization
ax = axes[1, 1]
ax.axis('off')

# Show context with highlighted answer
if results[0]['char_start'] != -1:
    before = context[:results[0]['char_start']]
    answer_text = results[0]['answer']
    after = context[results[0]['char_end']:]

    display_text = (
        f"Context (answer highlighted):\n\n"
        f"{before[:100]}...\n\n"
        f">>> {answer_text} <<<\n\n"
        f"...{after[:100]}"
    )
else:
    display_text = "Answer not found in context"

ax.text(0.1, 0.9, display_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.set_title('Answer Extraction Visualization')

plt.tight_layout()
plt.savefig('question_answering_system.png', dpi=150, bbox_inches='tight')
print("\n📊 Gráfico salvo: question_answering_system.png")

print("\n✅ Question answering demo completo!")
print("\n💡 QA SYSTEM TYPES:")
print("   - Extractive QA: Extract answer span from context (BERT, RoBERTa)")
print("   - Abstractive QA: Generate answer (T5, BART, GPT)")
print("   - Open-domain QA: No context provided (RAG, Wikipedia search)")
print("\n💡 KEY COMPONENTS:")
print("   - Start/End logits: Probability distribution over tokens")
print("   - Confidence: Product of start and end probabilities")
print("   - Max length: Limit answer span length")
print("   - No-answer detection: Model can say 'unanswerable'")
print("\n💡 APPLICATIONS:")
print("   - Customer support chatbots")
print("   - Document search and retrieval")
print("   - FAQ systems")
print("   - Knowledge base querying")
