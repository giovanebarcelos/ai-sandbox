"""
GO1728 - Few-Shot Chain-of-Thought (CoT)
=========================================
Demonstra Few-Shot CoT: fornecer exemplos completos de raciocínio no prompt.
Requer apenas bibliotecas padrão.

Conceito: Few-Shot CoT (Wei et al. 2022) é mais poderoso que Zero-Shot CoT
porque ensina ao modelo O ESTILO de raciocínio desejado via exemplos.
O modelo aprende:
  1. Como estruturar os passos
  2. O nível de detalhe esperado
  3. O formato da resposta final

Isso é chamado de "in-context learning" — o modelo "aprende" dentro do contexto
do prompt sem atualizar seus pesos (sem fine-tuning).
"""


def construir_prompt_few_shot_cot(questao_nova: str) -> str:
    """
    Constrói um prompt Few-Shot CoT com exemplos de raciocínio.
    Os exemplos ensinam ao modelo o padrão de resposta esperado.
    """
    prompt = """Aqui estão alguns exemplos de raciocínio matemático:

Q: Se uma pizza tem 8 fatias e comemos 3, quantas sobram?
A: Vamos pensar:
   1) Pizza original: 8 fatias
   2) Comidas: 3 fatias
   3) Sobram: 8 - 3 = 5 fatias
   Resposta: 5 fatias

Q: João tem 20 reais. Ele ganha mais 15 reais e gasta 12. Quanto tem agora?
A: Vamos pensar:
   1) Inicial: 20 reais
   2) Ganha: +15 reais → 20 + 15 = 35 reais
   3) Gasta: -12 reais → 35 - 12 = 23 reais
   Resposta: 23 reais

Agora resolva esta:

Q: {questao}
A:"""

    return prompt.format(questao=questao_nova)


def resolver_com_raciocinio(questao: str, passos: list, resposta: str) -> dict:
    """
    Representa a solução estruturada que o LLM deve produzir.
    Retorna dicionário com prompt e solução esperada.
    """
    return {
        "questao": questao,
        "prompt": construir_prompt_few_shot_cot(questao),
        "raciocinio": passos,
        "resposta_final": resposta,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("GO1728 - FEW-SHOT CHAIN-OF-THOUGHT")
    print("=" * 60)

    # ─── Problema principal ───────────────────────────────────
    questao = "Maria tem 3 caixas, cada uma com 7 maçãs. Ela come 4 maçãs. Quantas sobram?"

    solucao = resolver_com_raciocinio(
        questao=questao,
        passos=[
            "Maria tem 3 caixas com 7 maçãs cada.",
            "Total de maçãs: 3 × 7 = 21 maçãs.",
            "Ela come 4 maçãs: 21 - 4 = 17 maçãs.",
        ],
        resposta="17 maçãs",
    )

    print("\nPROMPT FEW-SHOT CoT ENVIADO AO LLM:")
    print("─" * 60)
    print(solucao["prompt"])

    print("\nRESPOSTA ESPERADA DO LLM:")
    print("─" * 60)
    for i, passo in enumerate(solucao["raciocinio"], 1):
        print(f"   {i}) {passo}")
    print(f"   Resposta: {solucao['resposta_final']}")
    print()
    print("   O LLM SEGUE O PADRAO DOS EXEMPLOS!")

    # ─── Comparação Zero-Shot vs Few-Shot CoT ─────────────────
    print()
    print("─" * 60)
    print("ZERO-SHOT vs FEW-SHOT CoT")
    print("─" * 60)

    comparacao = [
        {
            "tipo": "Sem CoT",
            "prompt": questao,
            "resposta_tipica": "17 maçãs (pode variar)",
            "precisao": "~60% em problemas complexos",
        },
        {
            "tipo": "Zero-Shot CoT",
            "prompt": questao + "\n\nVamos pensar passo a passo:",
            "resposta_tipica": "Raciocínio parcial + resposta",
            "precisao": "~78% em problemas complexos",
        },
        {
            "tipo": "Few-Shot CoT",
            "prompt": f"[2 exemplos completos]\n\nQ: {questao}\nA:",
            "resposta_tipica": "Raciocínio estruturado seguindo exemplos",
            "precisao": "~87% em problemas complexos",
        },
    ]

    for comp in comparacao:
        print(f"\n  {comp['tipo']}:")
        print(f"    Precisao: {comp['precisao']}")
        print(f"    Resposta: {comp['resposta_tipica']}")

    # ─── Quando usar cada abordagem ───────────────────────────
    print()
    print("─" * 60)
    print("QUANDO USAR CADA ABORDAGEM?")
    print("─" * 60)
    print()
    print("  Sem CoT      -> Perguntas simples e diretas")
    print("  Zero-Shot    -> Problemas moderados sem exemplos disponíveis")
    print("  Few-Shot     -> Problemas complexos OU quando formato importa")
    print("  Fine-tuning  -> Domínios muito específicos (+ caro)")
    print()
    print("  Custo de tokens:")
    print("  Sem CoT: 1x | Zero-Shot CoT: 2x | Few-Shot CoT: 3-5x")
    print()
    print("  Referência: Wei et al. 2022 - 'Chain-of-Thought Prompting")
    print("  Elicits Reasoning in Large Language Models' (Google Brain)")
    print()
    print("  Ver GO1729 para implementação real com Ollama.")
