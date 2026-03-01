# GO1634-36hAvançadoMaskedLanguageModelingCom
from transformers import BertTokenizer, BertForMaskedLM, pipeline
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class BertMaskedLM:
    """Sistema de preenchimento de palavras mascaradas"""

    def __init__(self, model_name='bert-base-uncased'):
        print(f"🔄 Carregando {model_name}...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.model.eval()

        # Pipeline para facilitar
        self.fill_mask = pipeline('fill-mask', model=self.model, tokenizer=self.tokenizer)
        print("✅ Modelo carregado!")

    def predict_masked_word(self, text, top_k=10, visualize=True):
        """Prever palavra mascarada ([MASK])"""

        if '[MASK]' not in text:
            print("⚠️  Adicione [MASK] no texto onde quer predição")
            return None

        # Predição
        predictions = self.fill_mask(text, top_k=top_k)

        print(f"\n📝 Texto: {text}")
        print(f"\n🎯 Top-{top_k} predições:\n")

        for i, pred in enumerate(predictions, 1):
            print(f"{i:2d}. {pred['token_str']:15s} | Score: {pred['score']:.4f} | \"{pred['sequence']}\"")

        if visualize:
            self._visualize_predictions(text, predictions)

        return predictions

    def _visualize_predictions(self, text, predictions):
        """Visualizar predições"""

        tokens = [pred['token_str'].strip() for pred in predictions]
        scores = [pred['score'] for pred in predictions]

        plt.figure(figsize=(12, 8))

        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(tokens)))
        bars = plt.barh(range(len(tokens)), scores, color=colors, alpha=0.8, edgecolor='black')

        plt.yticks(range(len(tokens)), tokens, fontsize=11)
        plt.xlabel('Probability Score', fontsize=12)
        plt.title(f'BERT Masked Word Predictions\nText: "{text}"', fontsize=14, fontweight='bold')
        plt.xlim(0, max(scores) * 1.1)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)

        # Adicionar valores
        for i, (bar, score) in enumerate(zip(bars, scores)):
            plt.text(score, i, f' {score:.4f}', va='center', fontsize=10)

        plt.tight_layout()
        plt.savefig('bert_masked_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()

    def multiple_masks(self, text, num_masks=2, iterations=5):
        """Prever múltiplas máscaras iterativamente"""

        print(f"\n📝 Texto original: {text}")
        print(f"🔄 Preenchendo {num_masks} máscaras em {iterations} iterações\n")

        current_text = text
        mask_positions = [i for i, word in enumerate(text.split()) if word == '[MASK]']

        for iteration in range(iterations):
            predictions = self.fill_mask(current_text)
            top_pred = predictions[0]

            current_text = top_pred['sequence']
            print(f"Iteração {iteration+1}: {current_text}")

            # Verificar se ainda há máscaras
            if '[MASK]' not in current_text:
                break

        print(f"\n✅ Texto final: {current_text}")
        return current_text

    def cloze_test(self, sentences, mask_position='random'):
        """Criar e resolver teste de preenchimento (cloze test)"""

        print("\n" + "="*80)
        print("CLOZE TEST - Preenchimento de Lacunas")
        print("="*80 + "\n")

        results = []

        for idx, sentence in enumerate(sentences, 1):
            words = sentence.split()

            # Escolher posição para mascarar
            if mask_position == 'random':
                mask_idx = np.random.randint(1, len(words)-1)  # Evitar primeira/última
            else:
                mask_idx = mask_position

            # Palavra original
            original_word = words[mask_idx]

            # Criar sentença mascarada
            masked_words = words.copy()
            masked_words[mask_idx] = '[MASK]'
            masked_sentence = ' '.join(masked_words)

            # Predição
            predictions = self.fill_mask(masked_sentence, top_k=5)
            top_prediction = predictions[0]['token_str'].strip()

            # Verificar acerto
            correct = (top_prediction.lower() == original_word.lower())

            results.append({
                'sentence': sentence,
                'masked': masked_sentence,
                'original': original_word,
                'predicted': top_prediction,
                'correct': correct,
                'score': predictions[0]['score']
            })

            print(f"{idx}. Original: {sentence}")
            print(f"   Masked:   {masked_sentence}")
            print(f"   Prediction: {top_prediction} (Original: {original_word}) {'✅' if correct else '❌'}")
            print(f"   Score: {predictions[0]['score']:.4f}\n")

        # Estatísticas
        accuracy = sum(r['correct'] for r in results) / len(results)
        avg_score = np.mean([r['score'] for r in results])

        print(f"📊 RESULTADOS:")
        print(f"   Acurácia: {accuracy:.1%}")
        print(f"   Score médio: {avg_score:.4f}")

        return results

    def context_sensitivity(self, word, contexts):
        """Demonstrar sensibilidade ao contexto"""

        print(f"\n🔍 Testando sensibilidade ao contexto para: '{word}'\n")

        fig, ax = plt.subplots(figsize=(14, 8))

        all_predictions = []
        context_labels = []

        for idx, context in enumerate(contexts):
            masked_text = context.replace(word, '[MASK]')
            predictions = self.fill_mask(masked_text, top_k=10)

            # Verificar se palavra original está nas predições
            original_found = False
            original_rank = None

            for rank, pred in enumerate(predictions, 1):
                if pred['token_str'].strip().lower() == word.lower():
                    original_found = True
                    original_rank = rank
                    break

            print(f"{idx+1}. Context: {context}")
            print(f"   Masked:   {masked_text}")
            print(f"   Original '{word}' found: {'✅' if original_found else '❌'}" +
                  (f" at rank {original_rank}" if original_found else ""))
            print(f"   Top prediction: {predictions[0]['token_str']} ({predictions[0]['score']:.4f})\n")

            # Coletar para visualização
            tokens = [pred['token_str'].strip() for pred in predictions[:5]]
            scores = [pred['score'] for pred in predictions[:5]]
            all_predictions.append((tokens, scores))
            context_labels.append(f"Context {idx+1}")

        # Visualizar comparação
        x = np.arange(5)
        width = 0.15

        for idx, (tokens, scores) in enumerate(all_predictions):
            offset = width * (idx - len(all_predictions)/2 + 0.5)
            ax.bar(x + offset, scores, width, label=context_labels[idx], alpha=0.8)

        ax.set_ylabel('Probability Score')
        ax.set_xlabel('Top-5 Predictions')
        ax.set_title(f'Context Sensitivity Test - Original Word: "{word}"')
        ax.set_xticks(x)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('bert_context_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.show()

# Inicializar sistema
mlm = BertMaskedLM()

# Teste 1: Predição simples
print("\n" + "="*80)
print("TESTE 1: Predição de Palavra Mascarada")
print("="*80)

test_sentences = [
    "The capital of France is [MASK].",
    "I love to eat [MASK] for breakfast.",
    "Python is a popular programming [MASK].",
    "The [MASK] is shining brightly today.",
]

for sent in test_sentences:
    mlm.predict_masked_word(sent, top_k=8)

# Teste 2: Múltiplas máscaras
print("\n" + "="*80)
print("TESTE 2: Múltiplas Máscaras")
print("="*80)

multi_mask = "The [MASK] is a large [MASK] that lives in the ocean."
mlm.multiple_masks(multi_mask, num_masks=2, iterations=3)

# Teste 3: Cloze Test
print("\n" + "="*80)
print("TESTE 3: Cloze Test Automático")
print("="*80)

cloze_sentences = [
    "Machine learning is a subset of artificial intelligence.",
    "The quick brown fox jumps over the lazy dog.",
    "Python is an interpreted high-level programming language.",
    "Neural networks are inspired by the human brain.",
    "Deep learning requires large amounts of data.",
]

results = mlm.cloze_test(cloze_sentences, mask_position='random')

# Teste 4: Sensibilidade ao contexto
print("\n" + "="*80)
print("TESTE 4: Sensibilidade ao Contexto")
print("="*80)

word = "bank"
contexts = [
    "I need to deposit money at the bank.",  # Instituição financeira
    "We sat on the bank of the river.",      # Margem do rio
    "The plane made a steep bank to the left.",  # Inclinação
]

mlm.context_sensitivity(word, contexts)

print("\n✅ Análise completa!")
print("\n📊 APLICAÇÕES:")
print("   - Correção automática de texto")
print("   - Sugestões de preenchimento")
print("   - Testes de compreensão")
print("   - Análise de contexto semântico")
