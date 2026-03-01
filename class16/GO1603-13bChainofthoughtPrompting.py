# GO1603-13bChainofthoughtPrompting
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import json

class ChainOfThoughtDemo:
    """
    Chain-of-Thought (CoT) Prompting

    Técnica para melhorar raciocínio de LLMs:
    - Descomposição do problema
    - Passos intermediários explícitos
    - Verificação de cada etapa

    Types:
    - Zero-shot CoT: "Let's think step by step"
    - Few-shot CoT: Exemplos com raciocínio
    - Self-consistency: Múltiplas chains
    """

    def __init__(self):
        pass

    def standard_prompting(self, question: str) -> str:
        """
        Standard prompting (baseline)

        Direct question → Direct answer
        """
        # Simulated response
        if "age" in question.lower():
            return "42"  # May be wrong due to no reasoning
        return "Answer"

    def zero_shot_cot(self, question: str) -> Dict:
        """
        Zero-shot CoT: Add "Let's think step by step"

        Triggers reasoning without examples
        """
        enhanced_prompt = question + "\nLet's think step by step:"

        # Simulated reasoning
        if "age" in question.lower() and "twice" in question.lower():
            reasoning = """
Step 1: John is currently 15 years old.
Step 2: In 5 years, John will be 15 + 5 = 20 years old.
Step 3: Mary is twice as old as John, so Mary is currently 15 × 2 = 30 years old.
Step 4: In 5 years, Mary will be 30 + 5 = 35 years old.
Answer: Mary will be 35 years old.
"""
            answer = "35"
        else:
            reasoning = "Step 1: Analyze the question...\n"
            answer = "Answer"

        return {
            'prompt': enhanced_prompt,
            'reasoning': reasoning,
            'answer': answer
        }

    def few_shot_cot(self, question: str, examples: List[Dict]) -> Dict:
        """
        Few-shot CoT: Provide examples with reasoning

        Shows model how to reason
        """
        # Build prompt with examples
        prompt = "Solve the following problems step by step:\n\n"

        for ex in examples:
            prompt += f"Q: {ex['question']}\n"
            prompt += f"A: {ex['reasoning']}\n"
            prompt += f"Answer: {ex['answer']}\n\n"

        prompt += f"Q: {question}\nA:"

        # Simulated response
        response = self.zero_shot_cot(question)

        return {
            'prompt': prompt,
            'reasoning': response['reasoning'],
            'answer': response['answer']
        }

    def self_consistency(self, question: str, num_samples: int = 5) -> Dict:
        """
        Self-consistency: Generate multiple reasoning paths

        Take majority vote on final answers
        """
        answers = []
        reasonings = []

        for i in range(num_samples):
            result = self.zero_shot_cot(question)
            answers.append(result['answer'])
            reasonings.append(result['reasoning'])

        # Majority vote
        from collections import Counter
        answer_counts = Counter(answers)
        final_answer = answer_counts.most_common(1)[0][0]
        confidence = answer_counts[final_answer] / num_samples

        return {
            'answers': answers,
            'reasonings': reasonings,
            'final_answer': final_answer,
            'confidence': confidence
        }

    def compare_methods(self, question: str) -> Dict:
        """Compare all CoT methods"""
        results = {}

        # Standard
        results['standard'] = {
            'method': 'Standard Prompting',
            'answer': self.standard_prompting(question),
            'reasoning_steps': 0,
            'accuracy': 0.45  # Simulated
        }

        # Zero-shot CoT
        zero_shot_result = self.zero_shot_cot(question)
        results['zero_shot_cot'] = {
            'method': 'Zero-shot CoT',
            'answer': zero_shot_result['answer'],
            'reasoning_steps': len(zero_shot_result['reasoning'].split('\n')),
            'accuracy': 0.72  # Simulated
        }

        # Few-shot CoT
        examples = [
            {
                'question': 'If x=10 and y=20, what is x+y?',
                'reasoning': 'Step 1: x = 10\nStep 2: y = 20\nStep 3: x + y = 10 + 20 = 30',
                'answer': '30'
            }
        ]
        few_shot_result = self.few_shot_cot(question, examples)
        results['few_shot_cot'] = {
            'method': 'Few-shot CoT',
            'answer': few_shot_result['answer'],
            'reasoning_steps': len(few_shot_result['reasoning'].split('\n')),
            'accuracy': 0.85  # Simulated
        }

        # Self-consistency
        self_cons_result = self.self_consistency(question)
        results['self_consistency'] = {
            'method': 'Self-Consistency CoT',
            'answer': self_cons_result['final_answer'],
            'reasoning_steps': 5,  # Multiple samples
            'accuracy': 0.91  # Simulated
        }

        return results

# === DEMO ===

print("🔁 Chain-of-Thought Prompting\n")
print("="*70)

demo = ChainOfThoughtDemo()

# Example problem
question = "John is 15 years old. Mary is twice as old as John. How old will Mary be in 5 years?"

print(f"\n📌 Problem:\n{question}\n")

# Standard prompting
print("\n1️⃣ STANDARD PROMPTING:\n")
answer_standard = demo.standard_prompting(question)
print(f"Answer: {answer_standard}")
print("❌ Incorrect! No reasoning shown.")

# Zero-shot CoT
print("\n\n2️⃣ ZERO-SHOT COT:\n")
result_zero = demo.zero_shot_cot(question)
print("Prompt: " + question + "\nLet's think step by step:")
print(f"\n{result_zero['reasoning']}")
print(f"✅ Correct! Shows clear reasoning.")

# Few-shot CoT
print("\n\n3️⃣ FEW-SHOT COT:\n")
examples = [
    {
        'question': 'Tom has 5 apples. He buys 3 more. How many does he have?',
        'reasoning': 'Step 1: Tom starts with 5 apples\nStep 2: He buys 3 more\nStep 3: Total = 5 + 3 = 8',
        'answer': '8'
    }
]

result_few = demo.few_shot_cot(question, examples)
print("(Shows example, then asks question)")
print(f"\n{result_few['reasoning']}")

# Self-consistency
print("\n\n4️⃣ SELF-CONSISTENCY COT:\n")
result_self = demo.self_consistency(question, num_samples=5)
print("Generated 5 reasoning paths:")
for i, ans in enumerate(result_self['answers'], 1):
    print(f"   Path {i}: {ans}")
print(f"\nMajority vote: {result_self['final_answer']}")
print(f"Confidence: {result_self['confidence']:.0%}")

# Compare all methods
print("\n\n📊 COMPARISON:\n")

comparison = demo.compare_methods(question)

print(f"{'Method':<25} {'Answer':<10} {'Reasoning Steps':<18} {'Accuracy'}")
print("-" * 70)

for key, data in comparison.items():
    print(f"{data['method']:<25} {data['answer']:<10} {data['reasoning_steps']:<18} {data['accuracy']:.0%}")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Accuracy comparison
ax = axes[0, 0]

methods = ['Standard', 'Zero-shot\nCoT', 'Few-shot\nCoT', 'Self-\nConsistency']
accuracies = [0.45, 0.72, 0.85, 0.91]
colors_acc = ['red', 'yellow', 'lightgreen', 'green']

bars = ax.bar(methods, accuracies, color=colors_acc, alpha=0.7)
ax.set_ylabel('Accuracy')
ax.set_title('Reasoning Task Accuracy by Method')
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)

for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{acc:.0%}', ha='center', va='bottom', fontweight='bold')

# 2. Task type performance
ax = axes[0, 1]

task_types = ['Arithmetic', 'Logic', 'Commonsense', 'Symbolic']
standard_scores = [0.35, 0.40, 0.55, 0.30]
cot_scores = [0.85, 0.78, 0.82, 0.70]

x = np.arange(len(task_types))
width = 0.35

ax.bar(x - width/2, standard_scores, width, label='Standard', alpha=0.8, color='lightcoral')
ax.bar(x + width/2, cot_scores, width, label='CoT', alpha=0.8, color='lightgreen')

ax.set_ylabel('Accuracy')
ax.set_title('CoT Performance by Task Type')
ax.set_xticks(x)
ax.set_xticklabels(task_types)
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1)

# 3. Model size vs CoT benefit
ax = axes[1, 0]

model_sizes = [1, 10, 100, 1000]  # Billion parameters (log scale)
cot_benefit = [0.05, 0.15, 0.30, 0.45]  # Accuracy improvement

ax.plot(model_sizes, cot_benefit, 'o-', linewidth=2, markersize=10, color='purple')
ax.fill_between(model_sizes, 0, cot_benefit, alpha=0.3, color='purple')
ax.set_xlabel('Model Size (B parameters)')
ax.set_ylabel('CoT Accuracy Improvement')
ax.set_title('CoT Benefit Increases with Model Size')
ax.set_xscale('log')
ax.grid(alpha=0.3)

# Annotate emergence
ax.annotate('Emergence of\nReasoning', xy=(100, 0.30), xytext=(10, 0.35),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, fontweight='bold')

# 4. Self-consistency improvement
ax = axes[1, 1]

num_samples_list = [1, 3, 5, 10, 20, 40]
accuracy_improvement = [0.72, 0.80, 0.85, 0.89, 0.91, 0.92]
compute_cost = [1, 3, 5, 10, 20, 40]  # Relative cost

ax2 = ax.twinx()

line1 = ax.plot(num_samples_list, accuracy_improvement, 'o-', color='green', 
                linewidth=2, markersize=8, label='Accuracy')
line2 = ax2.plot(num_samples_list, compute_cost, 's-', color='red', 
                 linewidth=2, markersize=8, label='Cost')

ax.set_xlabel('Number of Samples (Self-Consistency)')
ax.set_ylabel('Accuracy', color='green')
ax2.set_ylabel('Compute Cost (relative)', color='red')
ax.set_title('Self-Consistency: Accuracy vs Cost')
ax.tick_params(axis='y', labelcolor='green')
ax2.tick_params(axis='y', labelcolor='red')
ax.grid(alpha=0.3)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='center right')

# Mark sweet spot
ax.plot(5, 0.85, 'y*', markersize=20)
ax.annotate('Sweet Spot', xy=(5, 0.85), xytext=(15, 0.80),
            arrowprops=dict(arrowstyle='->', color='black', lw=2),
            fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('chain_of_thought.png', dpi=150, bbox_inches='tight')
print("\n\n📊 Gráfico salvo: chain_of_thought.png")

print("\n\n✅ Chain-of-Thought prompting implementado!")
print("\n💡 WHEN TO USE:")
print("   - Arithmetic/math problems")
print("   - Multi-step reasoning")
print("   - Logic puzzles")
print("   - Commonsense reasoning")
print("\n💡 BEST PRACTICES:")
print("   - Zero-shot CoT: Add 'Let's think step by step'")
print("   - Few-shot CoT: Provide 2-3 examples with reasoning")
print("   - Self-consistency: Use 5-10 samples for important tasks")
print("   - Works best with models ≥100B parameters")
