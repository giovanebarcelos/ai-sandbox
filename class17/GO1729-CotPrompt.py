"""
GO1729 - Chain-of-Thought com Ollama (implementação real)
==========================================================
Implementa CoT Zero-Shot e Few-Shot usando Ollama como backend LLM.
Compara as duas abordagens na mesma questão.

Instalação:
    pip install ollama
    ollama pull llama3.2

Execução:
    Iniciar: ollama serve
    Rodar:   python GO1729-CotPrompt.py

Conceito: este arquivo implementa o conceito de GO1726-GO1728 com
um LLM real. Observamos que Few-Shot CoT produz raciocínio mais
estruturado e respostas mais confiáveis que Zero-Shot CoT.
"""

import sys


def cot_prompt(question: str, method: str = "zero-shot") -> str:
    """
    Chain-of-Thought prompting com Ollama.

    method: "zero-shot" ou "few-shot"
    Retorna a resposta do modelo.
    """
    try:
        import ollama
    except ImportError:
        raise RuntimeError("ollama não instalado. Execute: pip install ollama")

    if method == "zero-shot":
        prompt = f"""Questão: {question}

Vamos pensar passo a passo para resolver isso:
"""

    elif method == "few-shot":
        prompt = f"""Aqui estão exemplos de raciocínio estruturado:

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
    else:
        raise ValueError(f"method deve ser 'zero-shot' ou 'few-shot', recebeu '{method}'")

    response = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': prompt}],
    )
    return response['message']['content']


def calcular_desconto_cascata(preco: float, d1: float, d2: float) -> dict:
    """
    Calcula desconto em cascata (verificação da resposta correta).
    d1, d2: percentuais (ex: 0.20 para 20%)
    """
    preco_d1 = preco * (1 - d1)
    preco_final = preco_d1 * (1 - d2)
    return {
        "preco_inicial": preco,
        "desconto_1": d1 * 100,
        "preco_apos_d1": preco_d1,
        "desconto_2": d2 * 100,
        "preco_final": preco_final,
    }


def demo_sem_ollama() -> None:
    """Mostra o que o script faria com o Ollama disponível."""
    questao = """Uma loja oferece 20% de desconto em um produto de R$ 250.
Depois aplica mais 10% sobre o valor com desconto.
Qual o preço final?"""

    info = calcular_desconto_cascata(250.0, 0.20, 0.10)

    print("=" * 70)
    print("GO1729 - CHAIN-OF-THOUGHT COM OLLAMA")
    print("=" * 70)
    print()
    print("Ollama nao disponivel. Mostrando simulacao da saida esperada.")
    print()
    print("QUESTAO:")
    print(questao)

    print()
    print("─" * 70)
    print("ZERO-SHOT CoT — prompt: 'Questão: ... \\n\\nVamos pensar passo a passo:'")
    print("─" * 70)
    print(f"[Saída esperada do LLM]")
    print(f"  Passo 1: Desconto de {info['desconto_1']:.0f}% em R$ {info['preco_inicial']:.2f}")
    print(f"           {info['desconto_1']:.0f}% de {info['preco_inicial']:.2f} = "
          f"R$ {info['preco_inicial'] - info['preco_apos_d1']:.2f}")
    print(f"           Preço após 1° desconto = R$ {info['preco_apos_d1']:.2f}")
    print(f"  Passo 2: Desconto de {info['desconto_2']:.0f}% sobre R$ {info['preco_apos_d1']:.2f}")
    print(f"           {info['desconto_2']:.0f}% de {info['preco_apos_d1']:.2f} = "
          f"R$ {info['preco_apos_d1'] - info['preco_final']:.2f}")
    print(f"  Passo 3: Preço final = R$ {info['preco_final']:.2f}")
    print(f"  Resposta: R$ {info['preco_final']:.2f}  ✓")

    print()
    print("─" * 70)
    print("FEW-SHOT CoT — prompt com 2 exemplos + questão")
    print("─" * 70)
    print("[Saída similar ao Zero-Shot, mas mais estruturada por seguir os exemplos]")
    print(f"  A resposta final é a mesma: R$ {info['preco_final']:.2f}")
    print("  Diferença: Few-Shot CoT produz raciocínio mais consistente e formatado.")

    print()
    print("─" * 70)
    print("IMPORTANTE: 20% + 10% ≠ 30% de desconto!")
    print("─" * 70)
    desconto_30 = 250 * (1 - 0.30)
    print(f"  Desconto simples 30% em R$ 250 = R$ {desconto_30:.2f}")
    print(f"  Cascata 20% + 10%:               R$ {info['preco_final']:.2f}")
    print(f"  Diferença: R$ {desconto_30 - info['preco_final']:.2f} "
          f"(o segundo desconto é sobre valor menor!)")

    print()
    print("Para usar com Ollama real:")
    print("  1. pip install ollama")
    print("  2. ollama pull llama3.2")
    print("  3. ollama serve")
    print("  4. python GO1729-CotPrompt.py")


# ─────────────────────────────────────────────────────────────
# RESULTADO ESPERADO (documentado aqui para referência)
# ─────────────────────────────────────────────────────────────
# Zero-Shot:
#   Passo 1: Desconto de 20% em R$ 250
#            20% de 250 = 0.20 × 250 = R$ 50
#            Preço após 1º desconto = 250 - 50 = R$ 200
#   Passo 2: Desconto de 10% em R$ 200
#            10% de 200 = 0.10 × 200 = R$ 20
#            Preço final = 200 - 20 = R$ 180
#   Resposta: R$ 180


if __name__ == "__main__":
    questao = """Uma loja oferece 20% de desconto em um produto de R$ 250.
Depois aplica mais 10% sobre o valor com desconto.
Qual o preço final?"""

    try:
        import ollama

        print("=" * 70)
        print("GO1729 - CoT COM OLLAMA (modelo real)")
        print("=" * 70)

        print("\n" + "─" * 70)
        print("ZERO-SHOT CoT")
        print("─" * 70)
        answer_zero = cot_prompt(questao, method="zero-shot")
        print(answer_zero)

        print("\n" + "─" * 70)
        print("FEW-SHOT CoT")
        print("─" * 70)
        answer_few = cot_prompt(questao, method="few-shot")
        print(answer_few)

        # Verificação
        info = calcular_desconto_cascata(250.0, 0.20, 0.10)
        print(f"\n  Verificacao matematica: R$ {info['preco_final']:.2f}")

    except ImportError:
        demo_sem_ollama()
    except Exception as e:
        print(f"Erro ao conectar ao Ollama: {e}")
        demo_sem_ollama()
