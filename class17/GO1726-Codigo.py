"""
GO1726 - Chain-of-Thought (CoT) Zero-Shot
==========================================
Demonstra a técnica CoT Zero-Shot: adicionar "Vamos pensar passo a passo"
ao prompt para induzir raciocínio explícito no LLM.
Requer apenas bibliotecas padrão.

Conceito: Zero-Shot CoT (Kojima et al. 2022) descobriu que a simples
adição da frase "Let's think step by step" ao prompt faz com que o modelo
gere etapas de raciocínio intermediárias antes da resposta, reduzindo
drasticamente erros em aritmética e lógica.

Diferença vs. Few-Shot CoT (GO1728):
- Zero-Shot CoT: sem exemplos, só o gatilho "passo a passo"
- Few-Shot CoT: fornece exemplos completos de raciocínio
"""


def construir_prompt_cot(pergunta: str) -> str:
    """
    Constrói prompt CoT Zero-Shot.
    A frase-gatilho força o modelo a "pensar em voz alta".
    """
    return f"{pergunta}\n\nVamos pensar passo a passo:"


def simular_resposta_cot(pergunta: str) -> str:
    """
    Simula a resposta de um LLM com CoT para o problema das bolas.
    Representa como o modelo DEVERIA raciocinar quando guiado.
    """
    if "bolas" in pergunta.lower() and "latas" in pergunta.lower():
        return (
            "Passo 1: Roger começa com 5 bolas de tênis.\n"
            "Passo 2: Ele compra 2 latas de bolas.\n"
            "Passo 3: Cada lata contém 3 bolas, então 2 × 3 = 6 bolas novas.\n"
            "Passo 4: Total = 5 (iniciais) + 6 (novas) = 11 bolas.\n"
            "Resposta: 11 bolas."
        )
    return "[Resposta gerada pelo LLM com raciocínio passo a passo]"


def comparar_sem_e_com_cot() -> None:
    """Mostra a diferença de resposta com e sem CoT."""
    pergunta = (
        "Roger tem 5 bolas de tênis. Ele compra mais 2 latas, "
        "cada uma com 3 bolas. Quantas bolas ele tem agora?"
    )

    print("─" * 60)
    print("PROMPT SEM CoT:")
    print("─" * 60)
    print(pergunta)
    print()
    print("Resposta do LLM: '10 bolas'  <- ERRADO! (erro típico)")
    print("  O modelo associa superficialmente os números sem multiplicar.")

    print()
    print("─" * 60)
    print("PROMPT COM CoT (adicionamos 'Vamos pensar passo a passo:'):")
    print("─" * 60)
    prompt_cot = construir_prompt_cot(pergunta)
    print(prompt_cot)
    print()
    print("Resposta do LLM:")
    print(simular_resposta_cot(pergunta))


def demonstrar_cot_outros_problemas() -> None:
    """Mostra o CoT em diferentes tipos de problemas."""

    problemas = [
        {
            "titulo": "Problema de velocidade",
            "pergunta": "Um trem viaja 300 km em 2 horas. Qual sua velocidade média?",
            "raciocinio": [
                "Distancia = 300 km",
                "Tempo = 2 horas",
                "Velocidade = Distancia / Tempo",
                "Velocidade = 300 / 2 = 150 km/h",
            ],
            "resposta": "150 km/h",
        },
        {
            "titulo": "Desconto em cascata",
            "pergunta": "Produto custa R$ 200. Desconto de 20%, depois mais 10%. Qual o preço final?",
            "raciocinio": [
                "Preco inicial = R$ 200",
                "1° desconto: 20% de 200 = 40 → 200 - 40 = R$ 160",
                "2° desconto: 10% de 160 = 16 → 160 - 16 = R$ 144",
                "Nota: 20% + 10% ≠ 30% (desconto sobre preço diferente)",
            ],
            "resposta": "R$ 144,00",
        },
    ]

    for prob in problemas:
        print("─" * 60)
        print(f"Problema: {prob['titulo']}")
        print("─" * 60)
        prompt = construir_prompt_cot(prob["pergunta"])
        print(f"Prompt:\n{prompt}\n")
        print("Resposta LLM com CoT:")
        for i, passo in enumerate(prob["raciocinio"], 1):
            print(f"  Passo {i}: {passo}")
        print(f"  Resposta final: {prob['resposta']}")
        print()


if __name__ == "__main__":
    print("=" * 60)
    print("GO1726 - CHAIN-OF-THOUGHT ZERO-SHOT")
    print("=" * 60)

    comparar_sem_e_com_cot()

    print()
    print("OUTROS EXEMPLOS DE CoT:")
    demonstrar_cot_outros_problemas()

    print("─" * 60)
    print("POR QUE CoT FUNCIONA?")
    print("─" * 60)
    print("  1. Gerar tokens de raciocínio ANTES da resposta guia")
    print("     o modelo por um caminho logicamente consistente.")
    print("  2. Cada passo 'ancora' o próximo (prompt aumenta contexto).")
    print("  3. Modelos maiores (>100B params) se beneficiam mais.")
    print()
    print("  Referência: Kojima et al. 2022 - 'Large Language Models are")
    print("  Zero-Shot Reasoners' (Google DeepMind)")
    print()
    print("  Veja GO1728 para Few-Shot CoT (com exemplos completos).")
