# GO1729-CotPrompt
import ollama

def cot_prompt(question, method="zero-shot"):
    """
    Chain-of-Thought prompting com Ollama
    """

    if method == "zero-shot":
        prompt = f"""
Questão: {question}

Vamos pensar passo a passo para resolver isso:
"""

    elif method == "few-shot":
        prompt = f"""
Aqui estão exemplos de raciocínio estruturado:

Exemplo 1:
Q: Se 3 notebooks custam R$ 150, quanto custa 1 notebook?
A: Vamos pensar:
   Passo 1: 3 notebooks = R$ 150
   Passo 2: 1 notebook = R$ 150 ÷ 3
   Passo 3: R$ 150 ÷ 3 = R$ 50
   Resposta: R$ 50

Exemplo 2:
Q: Um trem viaja 300 km em 2 horas. Qual a velocidade média?
A: Vamos pensar:
   Passo 1: Distância = 300 km
   Passo 2: Tempo = 2 horas
   Passo 3: Velocidade = Distância ÷ Tempo
   Passo 4: V = 300 ÷ 2 = 150 km/h
   Resposta: 150 km/h

Agora resolva:
Q: {question}
A:
"""

    response = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': prompt}]
    )

    return response['message']['content']

# ════════════════════════════════════════════════════════
# TESTAR CoT
# ════════════════════════════════════════════════════════

question = """
Uma loja oferece 20% de desconto em um produto de R$ 250.
Depois aplica mais 10% sobre o valor com desconto.
Qual o preço final?
"""

print("="*70)
print("ZERO-SHOT CoT")
print("="*70)
answer_zero = cot_prompt(question, method="zero-shot")
print(answer_zero)

print("\n" + "="*70)
print("FEW-SHOT CoT")
print("="*70)
answer_few = cot_prompt(question, method="few-shot")
print(answer_few)

# RESULTADO ESPERADO (Zero-Shot):
# Passo 1: Desconto de 20% em R$ 250
#          20% de 250 = 0.20 × 250 = R$ 50
#          Preço após 1º desconto = 250 - 50 = R$ 200
#
# Passo 2: Desconto de 10% em R$ 200
#          10% de 200 = 0.10 × 200 = R$ 20
#          Preço final = 200 - 20 = R$ 180
#
# Resposta: R$ 180
