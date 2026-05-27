# GO1603-13bChainofthoughtPrompting
import numpy as np
from typing import List, Dict
import json

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

class ChainOfThoughtDemo:
    """
    Chain-of-Thought (CoT) Prompting

    Técnica para melhorar raciocínio de LLMs:
    - Decomposição do problema
    - Passos intermediários explícitos
    - Verificação de cada etapa

    Tipos:
    - Zero-shot CoT: "Vamos pensar passo a passo"
    - Few-shot CoT: Exemplos com raciocínio
    - Self-consistency: Múltiplas chains
    """

    def __init__(self):
        pass

    def standard_prompting(self, question: str) -> str:
        """
        Prompting padrão (baseline)

        Pergunta direta → Resposta direta
        """
        # Resposta simulada
        if "age" in question.lower() or "anos" in question.lower():
            return "42"  # Pode errar por falta de raciocínio
        return "Resposta"

    def zero_shot_cot(self, question: str) -> Dict:
        """
        Zero-shot CoT: adiciona "Vamos pensar passo a passo"

        Dispara o raciocínio sem precisar de exemplos
        """
        enhanced_prompt = question + "\nVamos pensar passo a passo:"

        # Raciocínio simulado
        if ("anos" in question.lower() or "age" in question.lower()) and ("duas vezes" in question.lower() or "twice" in question.lower()):
            reasoning = """
Passo 1: João tem atualmente 15 anos.
Passo 2: Daqui a 5 anos, João terá 15 + 5 = 20 anos.
Passo 3: Maria tem o dobro da idade de João, então Maria tem atualmente 15 × 2 = 30 anos.
Passo 4: Daqui a 5 anos, Maria terá 30 + 5 = 35 anos.
Resposta: Maria terá 35 anos.
"""
            answer = "35"
        else:
            reasoning = "Passo 1: Analisar a pergunta...\n"
            answer = "Resposta"

        return {
            'prompt': enhanced_prompt,
            'reasoning': reasoning,
            'answer': answer
        }

    def few_shot_cot(self, question: str, examples: List[Dict]) -> Dict:
        """
        Few-shot CoT: fornece exemplos com raciocínio

        Mostra ao modelo como raciocinar
        """
        # Monta o prompt com exemplos
        prompt = "Resolva os problemas a seguir passo a passo:\n\n"

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
        Self-consistency: gera múltiplos caminhos de raciocínio

        Aplica voto majoritário nas respostas finais
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
        """Compara todos os métodos CoT"""
        results = {}

        # Padrão
        results['standard'] = {
            'method': 'Prompting Padrão',
            'answer': self.standard_prompting(question),
            'reasoning_steps': 0,
            'accuracy': 0.45  # Simulado
        }

        # Zero-shot CoT
        zero_shot_result = self.zero_shot_cot(question)
        results['zero_shot_cot'] = {
            'method': 'Zero-shot CoT',
            'answer': zero_shot_result['answer'],
            'reasoning_steps': len(zero_shot_result['reasoning'].split('\n')),
            'accuracy': 0.72  # Simulado
        }

        # Few-shot CoT
        examples = [
            {
                'question': 'Se x=10 e y=20, quanto é x+y?',
                'reasoning': 'Passo 1: x = 10\nPasso 2: y = 20\nPasso 3: x + y = 10 + 20 = 30',
                'answer': '30'
            }
        ]
        few_shot_result = self.few_shot_cot(question, examples)
        results['few_shot_cot'] = {
            'method': 'Few-shot CoT',
            'answer': few_shot_result['answer'],
            'reasoning_steps': len(few_shot_result['reasoning'].split('\n')),
            'accuracy': 0.85  # Simulado
        }

        # Self-consistency
        self_cons_result = self.self_consistency(question)
        results['self_consistency'] = {
            'method': 'Self-Consistency CoT',
            'answer': self_cons_result['final_answer'],
            'reasoning_steps': 5,  # Múltiplas amostras
            'accuracy': 0.91  # Simulado
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
ax.set_ylabel('Acurácia')
ax.set_title('Acurácia em Tarefas de Raciocínio por Método')
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)

for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{acc:.0%}', ha='center', va='bottom', fontweight='bold')

# 2. Task type performance
ax = axes[0, 1]

task_types = ['Aritmética', 'Lógica', 'Senso Comum', 'Simbólico']
standard_scores = [0.35, 0.40, 0.55, 0.30]
cot_scores = [0.85, 0.78, 0.82, 0.70]

x = np.arange(len(task_types))
width = 0.35

ax.bar(x - width/2, standard_scores, width, label='Padrão', alpha=0.8, color='lightcoral')
ax.bar(x + width/2, cot_scores, width, label='CoT', alpha=0.8, color='lightgreen')

ax.set_ylabel('Acurácia')
ax.set_title('Desempenho do CoT por Tipo de Tarefa')
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
ax.set_xlabel('Tamanho do Modelo (B parâmetros)')
ax.set_ylabel('Melhoria de Acurácia com CoT')
ax.set_title('Benefício do CoT Aumenta com o Tamanho do Modelo')
ax.set_xscale('log')
ax.grid(alpha=0.3)

# Anota emerêtncia
ax.annotate('Emergência do\nRaciocínio', xy=(100, 0.30), xytext=(10, 0.35),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, fontweight='bold')

# 4. Self-consistency improvement
ax = axes[1, 1]

num_samples_list = [1, 3, 5, 10, 20, 40]
accuracy_improvement = [0.72, 0.80, 0.85, 0.89, 0.91, 0.92]
compute_cost = [1, 3, 5, 10, 20, 40]  # Relative cost

ax2 = ax.twinx()

line1 = ax.plot(num_samples_list, accuracy_improvement, 'o-', color='green', 
                linewidth=2, markersize=8, label='Acurácia')
line2 = ax2.plot(num_samples_list, compute_cost, 's-', color='red', 
                 linewidth=2, markersize=8, label='Custo')

ax.set_xlabel('Número de Amostras (Self-Consistency)')
ax.set_ylabel('Acurácia', color='green')
ax2.set_ylabel('Custo Computacional (relativo)', color='red')
ax.set_title('Self-Consistency: Acurácia vs Custo')
ax.tick_params(axis='y', labelcolor='green')
ax2.tick_params(axis='y', labelcolor='red')
ax.grid(alpha=0.3)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='center right')

# Mark sweet spot
ax.plot(5, 0.85, 'y*', markersize=20)
ax.annotate('Ponto Ótimo', xy=(5, 0.85), xytext=(15, 0.80),
            arrowprops=dict(arrowstyle='->', color='black', lw=2),
            fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()
print("\n\n📊 Gráfico salvo: chain_of_thought.png")

print("\n\n✅ Chain-of-Thought prompting implementado!")
print("\n💡 QUANDO USAR:")
print("   - Problemas de aritmética/matemática")
print("   - Raciocínio em múltiplos passos")
print("   - Quebra-cabeças de lógica")
print("   - Raciocínio de senso comum")
print("\n💡 MELHORES PRÁTICAS:")
print("   - Zero-shot CoT: adicione 'Vamos pensar passo a passo'")
print("   - Few-shot CoT: forneça 2-3 exemplos com raciocínio")
print("   - Self-consistency: use 5-10 amostras para tarefas importantes")
print("   - Funciona melhor com modelos ≥100B parâmetros")
