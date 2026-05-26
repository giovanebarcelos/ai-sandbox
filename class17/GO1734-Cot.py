"""
GO1734 - Few-Shot CoT: Ensinando Cálculo por Exemplos
======================================================
Demonstra como Few-Shot CoT ensina o LLM a usar um método específico
de cálculo (decompor em dezena + unidade para adição/multiplicação).
Requer apenas bibliotecas padrão.

Conceito: ao ver exemplos de como DECOMPOR números antes de calcular,
o LLM "aprende" a usar o mesmo método na nova questão.
Isso é útil quando queremos que o modelo use uma abordagem particular
(ex: decomposição lugar-a-lugar vs algoritmo de coluna).
"""

from typing import List, Dict


# ──────────────────────────────────────────────────────────────
# 1. TEMPLATE FEW-SHOT COM MÉTODO DE DECOMPOSIÇÃO
# ──────────────────────────────────────────────────────────────

PROMPT_DECOMPOSICAO = """Exemplos com raciocínio por decomposição:

Q: 23 + 47 = ?
A: Vamos somar:
   Dezenas: 20 + 40 = 60
   Unidades: 3 + 7 = 10
   Total: 60 + 10 = 70
   Resposta: 70

Q: 15 × 4 = ?
A: Vamos calcular por decomposição:
   15 × 4 = (10 + 5) × 4
          = 10 × 4 + 5 × 4
          = 40 + 20
          = 60
   Resposta: 60

Agora resolva com o mesmo método:
Q: {questao}
A:"""


def construir_prompt(questao: str) -> str:
    return PROMPT_DECOMPOSICAO.format(questao=questao)


# ──────────────────────────────────────────────────────────────
# 2. SOLVER: demonstra os cálculos passo a passo
# ──────────────────────────────────────────────────────────────

def resolver_multiplicacao_decomposicao(a: int, b: int) -> Dict:
    """
    Resolve multiplicação via decomposição (método ensinado pelos exemplos).
    Modela a resposta esperada do LLM.
    """
    dezena_a = (a // 10) * 10
    unidade_a = a % 10

    parcela_1 = dezena_a * b
    parcela_2 = unidade_a * b
    total = parcela_1 + parcela_2

    return {
        "a": a,
        "b": b,
        "decomposicao": f"({dezena_a} + {unidade_a}) × {b}",
        "parcela_1": f"{dezena_a} × {b} = {parcela_1}",
        "parcela_2": f"{unidade_a} × {b} = {parcela_2}",
        "soma": f"{parcela_1} + {parcela_2}",
        "total": total,
    }


def resolver_soma_decomposicao(a: int, b: int) -> Dict:
    """Resolve soma via decomposição dezenas/unidades."""
    dezena_a, unidade_a = (a // 10) * 10, a % 10
    dezena_b, unidade_b = (b // 10) * 10, b % 10

    soma_dezenas = dezena_a + dezena_b
    soma_unidades = unidade_a + unidade_b
    total = soma_dezenas + soma_unidades

    return {
        "a": a,
        "b": b,
        "dezenas": f"{dezena_a} + {dezena_b} = {soma_dezenas}",
        "unidades": f"{unidade_a} + {unidade_b} = {soma_unidades}",
        "total": total,
    }


# ──────────────────────────────────────────────────────────────
# DEMONSTRAÇÃO
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("GO1734 - FEW-SHOT CoT: ENSINANDO METODO DE CALCULO")
    print("=" * 60)

    # ─── Problema-alvo do slide ───────────────────────────────
    questao_alvo = "34 × 12 = ?"
    print("\nPROMPT (template + questão-alvo):")
    print("─" * 60)
    print(construir_prompt(questao_alvo))

    # ─── Mostrar a solução esperada ───────────────────────────
    info = resolver_multiplicacao_decomposicao(34, 12)
    print("─" * 60)
    print("RESPOSTA ESPERADA DO LLM (seguindo o método dos exemplos):")
    print("─" * 60)
    print(f"  Vamos calcular por decomposição:")
    print(f"  {info['a']} × {info['b']} = {info['decomposicao']}")
    print(f"  = {info['parcela_1']}")
    print(f"  + {info['parcela_2']}")
    print(f"  = {info['soma']} = {info['total']}")
    print(f"  Resposta: {info['total']}")

    # ─── Mais exemplos ────────────────────────────────────────
    print()
    print("─" * 60)
    print("MAIS EXEMPLOS COM DECOMPOSIÇÃO:")
    print("─" * 60)

    exercicios_mult = [(25, 6), (47, 3), (18, 7)]
    print("\nMultiplicações:")
    for a, b in exercicios_mult:
        res = resolver_multiplicacao_decomposicao(a, b)
        print(f"  {a} × {b} = {res['decomposicao']}")
        print(f"           = {res['parcela_1']}")
        print(f"           + {res['parcela_2']}")
        print(f"           = {res['total']}")
        print()

    exercicios_soma = [(38, 54), (67, 25)]
    print("Somas:")
    for a, b in exercicios_soma:
        res = resolver_soma_decomposicao(a, b)
        print(f"  {a} + {b}: Dezenas = {res['dezenas']}")
        print(f"          Unidades = {res['unidades']}")
        print(f"          Total = {res['total']}")
        print()

    print("─" * 60)
    print("POR QUE FEW-SHOT ENSINA O METODO?")
    print("─" * 60)
    print("  O LLM aprende que decomposição é o padrão esperado.")
    print("  Sem exemplos: poderia usar cálculo direto (40×12 - 6×12).")
    print("  Com exemplos: usa especificamente decomposição dezena + unidade.")
    print()
    print("  Isso é útil quando queremos um método AUDITÁVEL e EXPLICÁVEL.")
    print("  Em finanças/medicina, o COMO calcular importa tanto quanto o QUE.")
