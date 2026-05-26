"""
GO1725 - O Problema dos LLMs sem Chain-of-Thought
==================================================
Demonstra por que LLMs "clássicos" erram em raciocínio em múltiplos passos.
Requer apenas bibliotecas padrão.

Conceito: LLMs treinados para predição de próximo token tendem a dar
respostas "rápidas" baseadas em padrões memorados. O problema "Roger e
as bolas de tênis" (Tennis Ball Problem) é um benchmark clássico que
revela essa falha — o modelo confunde 2 latas com 2 bolas e responde 9,
ou acerta por coincidência mas sem raciocínio real.

A solução é Chain-of-Thought (CoT): forçar o modelo a "pensar em voz alta"
passo a passo antes de dar a resposta final (ver GO1726).
"""


def calcular_bolas_correto() -> dict:
    """
    Calcula o problema das bolas de tênis passo a passo.
    Retorna a resposta correta com raciocínio explícito.
    """
    bolas_iniciais = 5
    latas_compradas = 2
    bolas_por_lata = 3
    bolas_novas = latas_compradas * bolas_por_lata  # 6
    total = bolas_iniciais + bolas_novas             # 11

    return {
        "bolas_iniciais": bolas_iniciais,
        "latas_compradas": latas_compradas,
        "bolas_por_lata": bolas_por_lata,
        "bolas_novas": bolas_novas,
        "total_correto": total,
        "resposta_errada_tipica": 10,  # Erro comum: 5 + 2*3 - 1 ou 5 + 5
    }


def simular_llm_sem_cot(pergunta: str) -> str:
    """
    Simula a resposta de um LLM sem Chain-of-Thought.
    Em problemas multi-step, LLMs tendem a fazer associações erradas.
    """
    # Padrão errado: LLM associa "2 latas" com "2 bolas" diretamente
    # Em vez de "2 latas × 3 bolas/lata = 6 bolas"
    respostas_tipicas_erradas = {
        "bolas": "10 bolas",        # Erro mais comum: 5 + 2 + 3 = 10
        "tênis": "9 bolas",         # Outro erro: confunde latas com bolas
    }
    for palavra, resposta in respostas_tipicas_erradas.items():
        if palavra in pergunta.lower():
            return resposta
    return "Não sei calcular isso."


if __name__ == "__main__":
    print("=" * 60)
    print("GO1725 - PROBLEMA DOS LLMs SEM CHAIN-OF-THOUGHT")
    print("=" * 60)

    # ─── O problema ───────────────────────────────────────────
    prompt = """
    Roger tem 5 bolas de tênis.
    Ele compra mais 2 latas de bolas,
    cada lata contendo 3 bolas.
    Quantas bolas ele tem agora?
    """

    print("\nProblema:")
    print(prompt)

    # ─── Resposta errada (LLM sem CoT) ────────────────────────
    resposta_errada = simular_llm_sem_cot(prompt)
    print(f"LLM SEM CoT responde: '{resposta_errada}'  <- ERRADO!")

    # ─── Por que erra? ────────────────────────────────────────
    print()
    print("Por que o LLM erra?")
    print("  1. Viu muitos textos com 'tem X + compra Y = X+Y'")
    print("  2. Faz associacao superficial: 5 bolas + 2 latas + 3 bolas")
    print("  3. Nao 'percebe' que precisa MULTIPLICAR latas × bolas/lata")
    print("  4. Prediz o token mais provavel, nao o correto")

    # ─── Resposta correta com raciocínio explícito ────────────
    info = calcular_bolas_correto()
    print()
    print("Raciocínio passo a passo (CORRETO):")
    print(f"  Passo 1: Roger começa com {info['bolas_iniciais']} bolas.")
    print(f"  Passo 2: Compra {info['latas_compradas']} latas.")
    print(f"  Passo 3: Cada lata tem {info['bolas_por_lata']} bolas.")
    print(f"  Passo 4: Bolas nas latas = {info['latas_compradas']} × {info['bolas_por_lata']}"
          f" = {info['bolas_novas']}")
    print(f"  Passo 5: Total = {info['bolas_iniciais']} + {info['bolas_novas']}"
          f" = {info['total_correto']} bolas  <- CORRETO!")

    # ─── Explicação do CoT ────────────────────────────────────
    print()
    print("Solucao: Chain-of-Thought (GO1726)")
    print("  Adicionar 'Vamos pensar passo a passo:' ao prompt")
    print("  forca o modelo a gerar tokens intermediarios de raciocínio")
    print("  antes de comprometer-se com a resposta final.")
    print()
    print("  Prompt CoT: '{pergunta}\\n\\nVamos pensar passo a passo:'")
    print("  LLM COM CoT responde: '11 bolas'  <- CORRETO!")
    print()
    print("  Referencia: Wei et al. 2022 - 'Chain-of-Thought Prompting")
    print("  Elicits Reasoning in Large Language Models' (Google Brain)")
