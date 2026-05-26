"""
GO1727 - Zero-Shot CoT com Frase-Gatilho
=========================================
Demonstra a técnica "Pense passo a passo" como gatilho automático.
Requer apenas bibliotecas padrão.

Conceito: A descoberta de Kojima et al. (2022) foi que simplesmente
adicionar "Let's think step by step" (em PT: "Vamos pensar passo a passo")
ao final do prompt é suficiente para induzir raciocínio estruturado.

Zero-Shot significa: sem exemplos de treinamento no prompt.
O modelo usa apenas o gatilho para saber que deve "raciocinar".

Problema de benchmark usado aqui: contagem de rodas no estacionamento.
"""


def construir_prompt_zero_shot_cot(questao: str) -> str:
    """
    Adiciona a frase-gatilho ao final da questão.
    'Pense passo a passo:' é o gatilho que ativa o raciocínio encadeado.
    """
    return f"Questão: {questao}\n\nPense passo a passo:"


def calcular_rodas_estacionamento(carros: int, motos: int) -> dict:
    """
    Calcula o número de rodas com raciocínio explícito.
    Modela o que o LLM deveria produzir com CoT.
    """
    rodas_carro = 4
    rodas_moto = 2

    total_carros = carros * rodas_carro
    total_motos = motos * rodas_moto
    total = total_carros + total_motos

    return {
        "carros": carros,
        "motos": motos,
        "rodas_por_carro": rodas_carro,
        "rodas_por_moto": rodas_moto,
        "total_rodas_carros": total_carros,
        "total_rodas_motos": total_motos,
        "total": total,
        "passsos": [
            f"Carros têm {rodas_carro} rodas cada. {carros} × {rodas_carro} = {total_carros} rodas.",
            f"Motos têm {rodas_moto} rodas cada. {motos} × {rodas_moto} = {total_motos} rodas.",
            f"Total = {total_carros} + {total_motos} = {total} rodas.",
        ],
    }


def simular_erro_sem_cot(carros: int, motos: int) -> int:
    """
    Simula erro típico de LLM sem CoT.
    Erro comum: somar os números diretamente sem multiplicar.
    """
    # Erro: 15 carros + 8 motos = 23 (ignora que cada veiculo tem N rodas)
    return carros + motos


if __name__ == "__main__":
    print("=" * 60)
    print("GO1727 - ZERO-SHOT CoT COM FRASE-GATILHO")
    print("=" * 60)

    carros = 15
    motos = 8
    questao = (f"Em um estacionamento há {carros} carros e {motos} motos. "
               "Quantas rodas há no total?")

    # ─── Sem CoT: erro provável ────────────────────────────────
    print("\nQuestao:")
    print(f"  {questao}")
    print()
    print("LLM SEM CoT:")
    resposta_errada = simular_erro_sem_cot(carros, motos)
    print(f"  '... há {resposta_errada} rodas no total.'  <- ERRADO!")
    print(f"  (Somou carros + motos = {carros} + {motos} = {resposta_errada}, ignorou rodas/veiculo)")

    # ─── Com Zero-Shot CoT ────────────────────────────────────
    print()
    print("PROMPT COM CoT:")
    prompt = construir_prompt_zero_shot_cot(questao)
    print(prompt)

    info = calcular_rodas_estacionamento(carros, motos)
    print()
    print("LLM COM CoT responde automaticamente:")
    for passo in info["passsos"]:
        print(f"  {passo}")
    print(f"  Resposta: {info['total']} rodas.  <- CORRETO!")

    # ─── Outros exemplos ─────────────────────────────────────
    print()
    print("─" * 60)
    print("MAIS EXEMPLOS DE PROBLEMAS MULTI-STEP:")
    print("─" * 60)

    exemplos = [
        {
            "questao": "Uma caixa tem 6 fileiras de ovos com 12 ovos cada. Quantos ovos total?",
            "passsos": [
                "Uma caixa: 6 fileiras × 12 ovos = 72 ovos.",
                "Resposta: 72 ovos.",
            ],
            "resposta": 72,
        },
        {
            "questao": "Se uma pizza tem 8 fatias e 4 amigos querem fatias iguais, quantas cada um recebe?",
            "passsos": [
                "Total de fatias: 8",
                "Número de amigos: 4",
                "Fatias por pessoa: 8 / 4 = 2 fatias.",
                "Resposta: 2 fatias cada.",
            ],
            "resposta": 2,
        },
    ]

    for ex in exemplos:
        print(f"\nQuestao: {ex['questao']}")
        print("Resposta com CoT:")
        for p in ex["passsos"]:
            print(f"  {p}")

    print()
    print("─" * 60)
    print("RESUMO:")
    print("─" * 60)
    print("  Zero-Shot CoT = adicionar 'Pense passo a passo:'")
    print("  Sem exemplos, apenas o gatilho linguístico.")
    print()
    print("  Quando usar Zero-Shot CoT?")
    print("  - Quando não temos exemplos de treinamento disponíveis")
    print("  - Quando o problema é relativamente simples")
    print("  - Como primeira tentativa antes de Few-Shot CoT")
    print()
    print("  Para problemas mais difíceis: Few-Shot CoT (GO1728)")
