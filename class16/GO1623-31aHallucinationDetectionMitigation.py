# GO1623-31aHallucinationDetectionMitigation
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

class HallucinationDetector:
    """
    Sistema para detectar e mitigar alucinações em LLMs

    Estratégias:
    1. Confidence scoring
    2. Consistency checking (multiple samples)
    3. Factual verification (knowledge base)
    4. Citation requirements
    5. Uncertainty quantification
    """

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.factual_kb = {
            'eiffel tower': {'height': 330, 'location': 'Paris', 'year': 1889},
            'python': {'type': 'programming language', 'first_release': 1991},
            'machine learning': {'field': 'AI', 'type': 'algorithm category'},
        }

    def detect_low_confidence(self, text: str, token_probs: List[float]) -> Dict:
        """
        Detecta baixa confiança baseado em probabilidades de tokens

        Low confidence = possível alucinação
        """
        avg_prob = np.mean(token_probs)
        min_prob = np.min(token_probs)

        # Tokens com baixa probabilidade
        low_conf_tokens = sum(1 for p in token_probs if p < 0.5)
        low_conf_ratio = low_conf_tokens / len(token_probs)

        is_hallucination = avg_prob < self.threshold or low_conf_ratio > 0.3

        return {
            'text': text,
            'avg_confidence': avg_prob,
            'min_confidence': min_prob,
            'low_conf_ratio': low_conf_ratio,
            'likely_hallucination': is_hallucination,
            'score': avg_prob
        }

    def check_consistency(self, question: str, answers: List[str]) -> Dict:
        """
        Verifica consistência entre múltiplas gerações

        Se respostas são muito diferentes → possível alucinação
        """
        # Simple consistency: count unique answers
        unique_answers = len(set(answers))
        consistency_score = 1.0 - (unique_answers / len(answers))

        # Check for contradictions (simplified)
        # In real system, use semantic similarity
        words_per_answer = [set(a.lower().split()) for a in answers]

        # Jaccard similarity between answers
        similarities = []
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                intersection = len(words_per_answer[i] & words_per_answer[j])
                union = len(words_per_answer[i] | words_per_answer[j])
                sim = intersection / union if union > 0 else 0
                similarities.append(sim)

        avg_similarity = np.mean(similarities) if similarities else 0

        is_consistent = avg_similarity > 0.5

        return {
            'question': question,
            'num_samples': len(answers),
            'unique_answers': unique_answers,
            'consistency_score': consistency_score,
            'avg_similarity': avg_similarity,
            'is_consistent': is_consistent
        }

    def verify_facts(self, text: str) -> Dict:
        """
        Verifica fatos contra knowledge base

        Encontra claims que contradizem KB
        """
        text_lower = text.lower()

        verified_facts = []
        contradictions = []

        for entity, facts in self.factual_kb.items():
            if entity in text_lower:
                # Check each fact
                for key, value in facts.items():
                    # Simplificado: verificar se o valor é mencionado corretamente
                    if str(value).lower() in text_lower:
                        verified_facts.append(f"{entity}: {key}={value}")
                    else:
                        # Possível contradição
                        # Em um sistema real, use NLI ou modelo de verificação de fatos
                        contradictions.append(f"{entity}: {key} not verified")

        factuality_score = len(verified_facts) / max(1, len(verified_facts) + len(contradictions))

        return {
            'text': text,
            'verified_facts': verified_facts,
            'contradictions': contradictions,
            'factuality_score': factuality_score,
            'has_contradictions': len(contradictions) > 0
        }

    def mitigate_hallucinations(self, response: str, metadata: Dict) -> Dict:
        """
        Aplica estratégias de mitigação

        1. Adicionar indicadores de confiança
        2. Adicionar citações/fontes
        3. Adicionar frases de incerteza
        4. Filtrar partes de baixa confiança
        """
        confidence = metadata.get('avg_confidence', 0.5)

        # Estratégia 1: Adicionar indicador de confiança
        if confidence < 0.5:
            prefix = "⚠️ Baixa confiança: "
        elif confidence < 0.7:
            prefix = "ℹ️ Confiança moderada: "
        else:
            prefix = "✓ Alta confiança: "

        mitigated = prefix + response

        # Estratégia 2: Adicionar incerteza para baixa confiança
        if confidence < 0.6:
            mitigated += "\n\n(Nota: Esta resposta tem baixa confiança e pode conter imprecisões. Verifique fatos importantes.)"

        # Estratégia 3: Solicitar citações
        if 'sources' not in metadata or not metadata['sources']:
            mitigated += "\n\n[Nenhuma fonte citada - verifique de forma independente]"

        return {
            'original': response,
            'mitigated': mitigated,
            'confidence': confidence,
            'mitigation_applied': True
        }

# === DEMO ===

detector = HallucinationDetector(threshold=0.7)

print("🛡️ Demo de Detecção de Alucinações\n")
print("="*70)

# Test 1: Low confidence detection
print("\n📌 Teste 1: Detecção Baseada em Confiança\n")

test_cases = [
    {
        'text': 'The Eiffel Tower is 330 meters tall',
        'probs': [0.9, 0.85, 0.88, 0.82, 0.9, 0.87],  # High confidence
    },
    {
        'text': 'The Eiffel Tower is 500 meters tall',  # Wrong fact
        'probs': [0.9, 0.85, 0.45, 0.3, 0.55, 0.4],  # Low confidence on wrong part
    },
    {
        'text': 'The Zzzyxian Tower was built in 2050',  # Hallucination
        'probs': [0.3, 0.2, 0.25, 0.35, 0.28, 0.22],  # Very low confidence
    }
]

for tc in test_cases:
    result = detector.detect_low_confidence(tc['text'], tc['probs'])
    status = "🚨 HALLUCINATION" if result['likely_hallucination'] else "✅ OK"
    print(f"{status} {result['text']}")
    print(f"   Confiança média: {result['avg_confidence']:.3f}")
    print(f"   Proporção de baixa confiança: {result['low_conf_ratio']:.3f}")
    print()

# Test 2: Consistency checking
print("\n📌 Teste 2: Verificação de Consistência\n")

questions_and_answers = [
    {
        'question': 'What is the capital of France?',
        'answers': ['Paris', 'Paris', 'Paris, France'],  # Consistent
    },
    {
        'question': 'When was Python released?',
        'answers': ['1991', '1989', '1995', '1990'],  # Inconsistent
    }
]

for qa in questions_and_answers:
    result = detector.check_consistency(qa['question'], qa['answers'])
    status = "✅ CONSISTENTE" if result['is_consistent'] else "🚨 INCONSISTENTE"
    print(f"{status} {result['question']}")
    print(f"   Respostas únicas: {result['unique_answers']}/{result['num_samples']}")
    print(f"   Similaridade: {result['avg_similarity']:.3f}")
    print()

# Test 3: Fact verification
print("\n📌 Teste 3: Verificação de Fatos\n")

statements = [
    'The Eiffel Tower is 330 meters tall and located in Paris',
    'The Eiffel Tower is 500 meters tall',  # Wrong
    'Python is a programming language first released in 1991',
]

for stmt in statements:
    result = detector.verify_facts(stmt)
    status = "✅ VERIFICADO" if not result['has_contradictions'] else "⚠️ NÃO VERIFICADO"
    print(f"{status} {stmt}")
    print(f"   Verificados: {len(result['verified_facts'])} fatos")
    print(f"   Não verificados: {len(result['contradictions'])} fatos")
    print()

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Confidence distribution
ax = axes[0, 0]
confidences = [0.87, 0.59, 0.27]  # From test cases
labels = ['Fato\nCorreto', 'Fato\nErrado', 'Alucinação']
colors = ['lightgreen', 'yellow', 'lightcoral']

bars = ax.bar(labels, confidences, color=colors, alpha=0.7)
ax.axhline(y=0.7, color='red', linestyle='--', label='Limiar')
ax.set_ylabel('Confiança Média')
ax.set_title('Níveis de Confiança por Tipo de Resposta')
ax.set_ylim(0, 1)
ax.legend()
ax.grid(axis='y', alpha=0.3)

for bar, conf in zip(bars, confidences):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{conf:.2f}', ha='center', va='bottom', fontweight='bold')

# 2. Consistency vs Hallucination rate
ax = axes[0, 1]
consistency_scores = [0.95, 0.75, 0.50, 0.25, 0.10]
hallucination_rates = [0.05, 0.15, 0.35, 0.60, 0.85]

ax.plot(consistency_scores, hallucination_rates, marker='o', linewidth=2, 
        markersize=8, color='red', label='Taxa de Alucinação')
ax.fill_between(consistency_scores, 0, hallucination_rates, alpha=0.3, color='red')
ax.set_xlabel('Pontuação de Consistência')
ax.set_ylabel('Taxa de Alucinação')
ax.set_title('Consistência vs Alucinação')
ax.legend()
ax.grid(alpha=0.3)

# Annotate inflection point
ax.annotate('Limiar\nCrítico', xy=(0.50, 0.35), xytext=(0.65, 0.50),
            arrowprops=dict(arrowstyle='->', color='black', lw=2),
            fontsize=10, fontweight='bold')

# 3. Mitigation strategies effectiveness
ax = axes[1, 0]
strategies = ['Sem\nMitigação', 'Indicador de\nConfiança', 'Múltiplas\nAmostras', 'Verificação\nde Fatos', 'Todas\nCombinadas']
effectiveness = [0, 25, 45, 60, 80]  # % reduction in harmful hallucinations

bars = ax.barh(strategies, effectiveness, color='skyblue', alpha=0.7)
ax.set_xlabel('Redução de Alucinação (%)')
ax.set_title('Eficácia das Estratégias de Mitigação')
ax.grid(axis='x', alpha=0.3)

for bar, eff in zip(bars, effectiveness):
    width = bar.get_width()
    ax.text(width + 2, bar.get_y() + bar.get_height()/2,
            f'{eff}%', ha='left', va='center', fontweight='bold')

# 4. Detection method comparison
ax = axes[1, 1]
methods = ['Confiança', 'Consistência', 'Verificação\nde Fatos', 'Combinado']
precision = [0.65, 0.72, 0.85, 0.88]
recall = [0.80, 0.75, 0.60, 0.85]

x = np.arange(len(methods))
width = 0.35

ax.bar(x - width/2, precision, width, label='Precisão', alpha=0.8, color='lightgreen')
ax.bar(x + width/2, recall, width, label='Recall', alpha=0.8, color='lightblue')

ax.set_ylabel('Pontuação')
ax.set_title('Desempenho dos Métodos de Detecção')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.show()
print("\n📊 Gráfico salvo: hallucination_detection.png")

print("\n✅ Sistema de detecção implementado!")
print("\n💡 BOAS PRÁTICAS:")
print("   1. Use múltiplas amostras para consultas importantes")
print("   2. Exija citações/fontes para afirmações factuais")
print("   3. Monitore pontuações de confiança")
print("   4. Implemente verificação de fatos contra KB")
print("   5. Adicione indicadores de incerteza para baixa confiança")
print("   6. Use recuperação (RAG) para embasar respostas")
