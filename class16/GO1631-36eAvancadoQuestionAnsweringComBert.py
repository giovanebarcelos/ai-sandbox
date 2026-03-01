# GO1631-36eAvançadoQuestionAnsweringComBert
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class BertQA:
    """Sistema de Question Answering com BERT"""

    def __init__(self, model_name='bert-large-uncased-whole-word-masking-finetuned-squad'):
        print(f"🔄 Carregando modelo: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForQuestionAnswering.from_pretrained(model_name)
        self.model.eval()
        print("✅ Modelo carregado!")

    def answer_question(self, question, context, visualize=True):
        """Responder pergunta dado um contexto"""

        # Tokenizar
        inputs = self.tokenizer(
            question,
            context,
            add_special_tokens=True,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            return_offsets_mapping=True
        )

        offset_mapping = inputs.pop('offset_mapping')[0]

        # Predição
        with torch.no_grad():
            outputs = self.model(**inputs)
            start_logits = outputs.start_logits[0]
            end_logits = outputs.end_logits[0]

        # Encontrar posição da resposta
        start_idx = torch.argmax(start_logits)
        end_idx = torch.argmax(end_logits)

        # Extrair resposta
        input_ids = inputs['input_ids'][0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        answer_tokens = tokens[start_idx:end_idx+1]
        answer = self.tokenizer.convert_tokens_to_string(answer_tokens)

        # Scores
        start_score = start_logits[start_idx].item()
        end_score = end_logits[end_idx].item()
        confidence = (start_score + end_score) / 2

        # Visualizar
        if visualize:
            self._visualize_prediction(
                tokens, start_logits, end_logits,
                start_idx, end_idx, question, answer
            )

        return {
            'answer': answer,
            'confidence': confidence,
            'start_idx': start_idx.item(),
            'end_idx': end_idx.item(),
            'start_score': start_score,
            'end_score': end_score
        }

    def _visualize_prediction(self, tokens, start_logits, end_logits, 
                             start_idx, end_idx, question, answer):
        """Visualizar scores de start/end"""

        # Limitar visualização aos primeiros 50 tokens
        max_len = min(50, len(tokens))
        tokens_vis = tokens[:max_len]
        start_logits_vis = start_logits[:max_len].numpy()
        end_logits_vis = end_logits[:max_len].numpy()

        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        # Start logits
        colors_start = ['red' if i == start_idx else 'blue' for i in range(max_len)]
        axes[0].bar(range(max_len), start_logits_vis, color=colors_start, alpha=0.6)
        axes[0].set_title(f'START Logits (max at position {start_idx})', fontsize=14)
        axes[0].set_ylabel('Logit Value')
        axes[0].set_xticks(range(max_len))
        axes[0].set_xticklabels(tokens_vis, rotation=90, fontsize=8)
        axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[0].grid(axis='y', alpha=0.3)

        # End logits
        colors_end = ['red' if i == end_idx else 'green' for i in range(max_len)]
        axes[1].bar(range(max_len), end_logits_vis, color=colors_end, alpha=0.6)
        axes[1].set_title(f'END Logits (max at position {end_idx})', fontsize=14)
        axes[1].set_ylabel('Logit Value')
        axes[1].set_xlabel('Token Position')
        axes[1].set_xticks(range(max_len))
        axes[1].set_xticklabels(tokens_vis, rotation=90, fontsize=8)
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[1].grid(axis='y', alpha=0.3)

        plt.suptitle(f'Question: "{question}"\nAnswer: "{answer}"', 
                    fontsize=12, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig('bert_qa_prediction.png', dpi=300, bbox_inches='tight')
        plt.show()

    def batch_qa(self, qa_pairs):
        """Responder múltiplas perguntas"""
        results = []

        for i, (question, context) in enumerate(qa_pairs, 1):
            print(f"\n{'='*70}")
            print(f"Q{i}: {question}")
            print(f"{'='*70}")

            result = self.answer_question(question, context, visualize=False)
            results.append(result)

            print(f"Answer: {result['answer']}")
            print(f"Confidence: {result['confidence']:.4f}")

        # Visualizar confiança
        self._plot_confidence(results, [q for q, _ in qa_pairs])

        return results

    def _plot_confidence(self, results, questions):
        """Plot de confiança das respostas"""
        confidences = [r['confidence'] for r in results]

        plt.figure(figsize=(12, 6))
        bars = plt.barh(range(len(questions)), confidences, color='skyblue', alpha=0.8)

        # Colorir por confiança
        for i, bar in enumerate(bars):
            if confidences[i] > 10:
                bar.set_color('green')
            elif confidences[i] > 5:
                bar.set_color('orange')
            else:
                bar.set_color('red')

        plt.yticks(range(len(questions)), 
                  [f"Q{i+1}: {q[:40]}..." for i, q in enumerate(questions)],
                  fontsize=9)
        plt.xlabel('Confidence Score', fontsize=12)
        plt.title('BERT Q&A Confidence Scores\n(Verde: Alta | Laranja: Média | Vermelho: Baixa)', 
                 fontsize=14)
        plt.grid(axis='x', alpha=0.3)

        for i, conf in enumerate(confidences):
            plt.text(conf, i, f' {conf:.2f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig('bert_qa_confidences.png', dpi=300, bbox_inches='tight')
        plt.show()

# Inicializar sistema
qa_system = BertQA()

# Contexto sobre IA
context_ai = """
Artificial Intelligence (AI) is intelligence demonstrated by machines, in contrast to 
the natural intelligence displayed by humans and animals. AI research has been defined 
as the field of study of intelligent agents, which refers to any system that perceives 
its environment and takes actions that maximize its chance of achieving its goals. 
The term artificial intelligence was coined in 1956 by John McCarthy at the Dartmouth 
Conference. Machine learning, a subset of AI, focuses on the development of algorithms 
that can learn from and make predictions on data. Deep learning, which is based on 
artificial neural networks, has led to breakthroughs in image recognition, natural 
language processing, and game playing. Modern AI systems can perform tasks such as 
visual perception, speech recognition, decision-making, and language translation.
"""

# Teste 1: Pergunta simples
print("\n" + "="*70)
print("TESTE 1: Pergunta Simples")
print("="*70)
result1 = qa_system.answer_question(
    "When was artificial intelligence coined?",
    context_ai
)

# Teste 2: Pergunta mais complexa
print("\n" + "="*70)
print("TESTE 2: Pergunta Complexa")
print("="*70)
result2 = qa_system.answer_question(
    "What is machine learning?",
    context_ai
)

# Batch de perguntas
print("\n" + "="*70)
print("TESTE 3: Batch de Perguntas")
print("="*70)

qa_pairs = [
    ("Who coined the term artificial intelligence?", context_ai),
    ("What is deep learning based on?", context_ai),
    ("What tasks can modern AI systems perform?", context_ai),
    ("What does AI research study?", context_ai),
]

results = qa_system.batch_qa(qa_pairs)

# Contexto adicional: Python
context_python = """
Python is a high-level, interpreted programming language created by Guido van Rossum 
and first released in 1991. Python's design philosophy emphasizes code readability 
with its notable use of significant whitespace. It provides constructs that enable 
clear programming on both small and large scales. Python supports multiple programming 
paradigms, including structured, object-oriented, and functional programming. It 
features a dynamic type system and automatic memory management.
"""

print("\n" + "="*70)
print("TESTE 4: Novo Contexto (Python)")
print("="*70)

result_python = qa_system.answer_question(
    "Who created Python?",
    context_python
)

print("\n✅ Análise completa!")
print("\n📊 RESUMO:")
print(f"   - BERT pré-treinado em SQuAD dataset")
print(f"   - Prediz START e END positions da resposta")
print(f"   - Confiança baseada em logits")
print(f"   - Funciona melhor quando resposta está no contexto")
