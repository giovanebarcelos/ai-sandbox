# GO1632-36fAvançadoTextGenerationComGpt2
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class ControlledTextGenerator:
    """Gerador de texto com GPT-2 e múltiplas estratégias"""

    def __init__(self, model_name='gpt2-medium'):
        print(f"🔄 Carregando {model_name}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()

        # Padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("✅ Modelo carregado!")

    def generate_greedy(self, prompt, max_length=50):
        """Greedy decoding: sempre escolhe token mais provável"""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                do_sample=False,  # Greedy
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def generate_sampling(self, prompt, max_length=50, temperature=1.0, top_k=0, top_p=1.0):
        """Sampling com temperatura e top-k/top-p"""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                top_k=top_k if top_k > 0 else None,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def generate_beam_search(self, prompt, max_length=50, num_beams=5):
        """Beam search: mantém top-k sequências"""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def compare_strategies(self, prompt, max_length=80):
        """Comparar diferentes estratégias"""

        strategies = {
            'Greedy': self.generate_greedy(prompt, max_length),
            'Temp=0.5': self.generate_sampling(prompt, max_length, temperature=0.5),
            'Temp=1.0': self.generate_sampling(prompt, max_length, temperature=1.0),
            'Temp=1.5': self.generate_sampling(prompt, max_length, temperature=1.5),
            'Top-k=50': self.generate_sampling(prompt, max_length, top_k=50),
            'Top-p=0.9': self.generate_sampling(prompt, max_length, top_p=0.9),
            'Beam=5': self.generate_beam_search(prompt, max_length, num_beams=5)
        }

        print(f"\n{'='*80}")
        print(f"PROMPT: \"{prompt}\"")
        print(f"{'='*80}\n")

        for strategy, text in strategies.items():
            print(f"🔹 {strategy}:")
            print(f"   {text[len(prompt):]}\n")

        return strategies

    def visualize_temperature_effect(self, prompt, num_samples=5):
        """Visualizar efeito da temperatura"""

        temperatures = [0.3, 0.7, 1.0, 1.3, 1.7]

        fig, axes = plt.subplots(len(temperatures), 1, figsize=(14, 12))

        for idx, temp in enumerate(temperatures):
            print(f"\n🌡️  Temperature = {temp}")

            # Gerar múltiplas amostras
            texts = []
            for i in range(num_samples):
                text = self.generate_sampling(prompt, max_length=30, temperature=temp)
                texts.append(text[len(prompt):])  # Remover prompt
                print(f"   Sample {i+1}: {text[len(prompt):][:50]}...")

            # Análise de diversidade: tokens únicos
            all_tokens = []
            for text in texts:
                tokens = text.split()
                all_tokens.extend(tokens)

            unique_ratio = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0
            token_counts = Counter(all_tokens)
            most_common = token_counts.most_common(10)

            # Plot
            if most_common:
                words, counts = zip(*most_common)
                axes[idx].barh(range(len(words)), counts, color='skyblue', alpha=0.8)
                axes[idx].set_yticks(range(len(words)))
                axes[idx].set_yticklabels(words, fontsize=9)
                axes[idx].set_xlabel('Frequency')
                axes[idx].set_title(f'Temp={temp} | Unique Tokens: {unique_ratio:.2%}', fontsize=11)
                axes[idx].invert_yaxis()
                axes[idx].grid(axis='x', alpha=0.3)

        plt.suptitle(f'Temperature Effect on Token Distribution\nPrompt: "{prompt}"', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('gpt2_temperature_effect.png', dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_next_token_probabilities(self, prompt, top_k=15):
        """Analisar probabilidades do próximo token"""

        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1, :]  # Último token
            probs = torch.softmax(logits, dim=0)

        # Top-k tokens
        top_probs, top_indices = torch.topk(probs, top_k)
        top_tokens = [self.tokenizer.decode([idx]) for idx in top_indices]

        # Visualizar
        plt.figure(figsize=(12, 8))
        plt.barh(range(top_k), top_probs.numpy(), color='coral', alpha=0.8)
        plt.yticks(range(top_k), top_tokens, fontsize=10)
        plt.xlabel('Probability', fontsize=12)
        plt.title(f'Top-{top_k} Next Token Predictions\nPrompt: "{prompt}"', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)

        for i, prob in enumerate(top_probs):
            plt.text(prob, i, f' {prob:.4f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig('gpt2_next_token_probs.png', dpi=300, bbox_inches='tight')
        plt.show()

        return list(zip(top_tokens, top_probs.numpy()))

# Inicializar gerador
generator = ControlledTextGenerator(model_name='gpt2')

# Teste 1: Comparar estratégias
print("\n" + "="*80)
print("TESTE 1: Comparação de Estratégias de Decodificação")
print("="*80)

prompt1 = "Artificial intelligence will"
strategies_result = generator.compare_strategies(prompt1, max_length=60)

# Teste 2: Efeito da temperatura
print("\n" + "="*80)
print("TESTE 2: Efeito da Temperatura")
print("="*80)

prompt2 = "In the future, robots will"
generator.visualize_temperature_effect(prompt2, num_samples=5)

# Teste 3: Probabilidades do próximo token
print("\n" + "="*80)
print("TESTE 3: Análise de Probabilidades do Próximo Token")
print("="*80)

prompt3 = "The president of the United States"
top_tokens = generator.analyze_next_token_probabilities(prompt3, top_k=20)

print("\n📊 Top-20 próximos tokens:")
for token, prob in top_tokens:
    print(f"   '{token}': {prob:.4f}")

# Teste 4: Geração criativa
print("\n" + "="*80)
print("TESTE 4: Geração Criativa (Alta Temperatura)")
print("="*80)

creative_prompts = [
    "Once upon a time in a magical forest,",
    "The secret to happiness is",
    "In the year 2050, technology will",
]

for prompt in creative_prompts:
    text = generator.generate_sampling(prompt, max_length=80, temperature=1.3, top_p=0.95)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {text[len(prompt):]}")

print("\n✅ Análise completa!")
print("\n📊 RESUMO DE ESTRATÉGIAS:")
print("   Greedy: Determinístico, sempre mesmo resultado")
print("   Temperature: Controla aleatoriedade (↑temp = ↑criatividade)")
print("   Top-k: Limita a k tokens mais prováveis")
print("   Top-p (nucleus): Considera tokens até prob cumulativa p")
print("   Beam Search: Mantém múltiplas hipóteses, melhor qualidade")
