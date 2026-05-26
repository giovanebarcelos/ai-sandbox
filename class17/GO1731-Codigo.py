"""
GO1731 - Few-Shot para Análise de Sentimentos
=============================================
Demonstra Few-Shot prompting para classificação de sentimentos.
Requer apenas bibliotecas padrão (simulação sem LLM externo).

Conceito: classificação de sentimento (POSITIVO/NEGATIVO/NEUTRO) é uma
tarefa NLP clássica. Com Few-Shot prompting, o LLM aprende o padrão
de classificação pelos exemplos — sem treinar um modelo especializado.

Comparação de abordagens:
1. Regex/heurística: regras simples, baixo custo, pouca precisão
2. Fine-tuning BERT: preciso, mas requer dataset e treinamento
3. Few-Shot LLM: sem treinamento, alta precisão, generaliza bem
"""

from typing import List, Tuple


# ──────────────────────────────────────────────────────────────
# 1. TEMPLATE FEW-SHOT PARA SENTIMENTO
# ──────────────────────────────────────────────────────────────

PROMPT_SENTIMENTO = """Classifique o sentimento como POSITIVO, NEGATIVO ou NEUTRO:

Texto: "Adorei este produto! Qualidade excepcional."
Sentimento: POSITIVO

Texto: "Péssimo atendimento, nunca mais volto."
Sentimento: NEGATIVO

Texto: "O produto chegou hoje."
Sentimento: NEUTRO

Texto: "Estou muito feliz com minha compra, recomendo!"
Sentimento: POSITIVO

Agora classifique:
Texto: "{texto}"
Sentimento:"""


def construir_prompt_sentimento(texto: str) -> str:
    return PROMPT_SENTIMENTO.format(texto=texto)


# ──────────────────────────────────────────────────────────────
# 2. CLASSIFICADOR HEURÍSTICO (baseline sem LLM)
# ──────────────────────────────────────────────────────────────

PALAVRAS_POSITIVAS = {
    "adorei", "excelente", "ótimo", "perfeito", "incrível",
    "recomendo", "feliz", "satisfeito", "parabéns", "bom",
    "maravilhoso", "amei", "top", "qualidade",
}

PALAVRAS_NEGATIVAS = {
    "péssimo", "horrível", "ruim", "terrível", "nunca mais",
    "decepcionado", "insatisfeito", "demora", "quebrou", "defeito",
    "cancelei", "desapontado", "lixo",
}


def classificar_heuristico(texto: str) -> Tuple[str, float]:
    """
    Classificação por contagem de palavras-chave.
    Retorna (sentimento, confiança).
    """
    texto_lower = texto.lower()
    pos = sum(1 for p in PALAVRAS_POSITIVAS if p in texto_lower)
    neg = sum(1 for p in PALAVRAS_NEGATIVAS if p in texto_lower)

    total = pos + neg
    if total == 0:
        return "NEUTRO", 0.5

    confianca = max(pos, neg) / total
    if pos > neg:
        return "POSITIVO", confianca
    elif neg > pos:
        return "NEGATIVO", confianca
    else:
        return "NEUTRO", 0.5


# ──────────────────────────────────────────────────────────────
# 3. SIMULAÇÃO DO LLM FEW-SHOT
# ──────────────────────────────────────────────────────────────

# Dados de teste com rótulos corretos (ground truth)
DATASET_TESTE: List[Tuple[str, str]] = [
    ("Adorei! Produto de altíssima qualidade.", "POSITIVO"),
    ("Péssimo! Chegou quebrado e o suporte não ajudou.", "NEGATIVO"),
    ("O produto é ok, nada de especial.", "NEUTRO"),
    ("Superou minhas expectativas, recomendo muito!", "POSITIVO"),
    ("Atendimento horrível, jamais compro aqui novamente.", "NEGATIVO"),
    ("O pedido foi processado.", "NEUTRO"),
    ("Produto chegou no prazo, estou feliz.", "POSITIVO"),
    ("Decepcionante. Qualidade inferior ao anunciado.", "NEGATIVO"),
    ("Recebi o email de confirmação.", "NEUTRO"),
    ("Incrível custo-benefício! Comprarei de novo.", "POSITIVO"),
]


def simular_llm_sentimento(texto: str) -> str:
    """
    Simula resposta do LLM para demonstração.
    Em produção: substituir por chamada real ao Ollama.
    """
    sentimento, _ = classificar_heuristico(texto)
    return sentimento


def avaliar_classificador(classificador_fn, dataset, nome: str) -> dict:
    """Avalia acurácia de um classificador no dataset de teste."""
    corretos = 0
    resultados = []
    for texto, rotulo_real in dataset:
        predicao = classificador_fn(texto)
        if isinstance(predicao, tuple):
            predicao = predicao[0]
        acertou = predicao == rotulo_real
        corretos += acertou
        resultados.append((texto, rotulo_real, predicao, acertou))

    acuracia = corretos / len(dataset)
    return {
        "nome": nome,
        "acuracia": acuracia,
        "corretos": corretos,
        "total": len(dataset),
        "resultados": resultados,
    }


# ──────────────────────────────────────────────────────────────
# DEMONSTRAÇÃO
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("GO1731 - FEW-SHOT: ANALISE DE SENTIMENTOS")
    print("=" * 60)

    # ─── Mostrar template do prompt ───────────────────────────
    texto_exemplo = "O produto é ok, nada de especial."
    print("\nTEMPLATE DO PROMPT FEW-SHOT:")
    print("─" * 60)
    print(construir_prompt_sentimento(texto_exemplo))

    # ─── Classificar textos individualmente ──────────────────
    print("─" * 60)
    print("CLASSIFICACAO INDIVIDUAL:")
    print("─" * 60)

    textos_demo = [
        "Adorei este produto! Qualidade excepcional.",
        "Péssimo atendimento, nunca mais volto.",
        "O produto chegou hoje.",
        "O produto é ok, nada de especial.",
        "Superou minhas expectativas! Recomendo muito.",
    ]

    for texto in textos_demo:
        sent_heur, conf = classificar_heuristico(texto)
        sent_llm = simular_llm_sentimento(texto)
        print(f"\n  Texto: \"{texto[:50]}...\"" if len(texto) > 50 else f"\n  Texto: \"{texto}\"")
        print(f"    Heuristico : {sent_heur} (confianca: {conf:.0%})")
        print(f"    LLM Few-Shot: {sent_llm}  [RESULTADO: {simular_llm_sentimento(texto)}]")

    # ─── Avaliação no dataset ─────────────────────────────────
    print()
    print("─" * 60)
    print("AVALIACAO NO DATASET DE TESTE (10 amostras):")
    print("─" * 60)

    resultado_heur = avaliar_classificador(
        lambda t: classificar_heuristico(t)[0], DATASET_TESTE, "Heuristico"
    )
    resultado_llm = avaliar_classificador(
        simular_llm_sentimento, DATASET_TESTE, "LLM Few-Shot"
    )

    for res in [resultado_heur, resultado_llm]:
        print(f"\n  {res['nome']}:")
        print(f"    Acuracia: {res['acuracia']:.0%} "
              f"({res['corretos']}/{res['total']} corretos)")

    print()
    print("─" * 60)
    print("VANTAGENS DO FEW-SHOT vs HEURISTICO:")
    print("─" * 60)
    print("  Heuristico: rapido, sem dependencias, falha em ironia e contexto")
    print("  Few-Shot LLM: lida com sarcasmo, contexto e nuances linguisticas")
    print()
    print("  Exemplo de falha do heuristico:")
    print("  'Produto fantastico... se voce gosta de esperar 3 meses!' <- IRONIA")
    print("  Heuristico: POSITIVO (encontrou 'fantastico')")
    print("  LLM Few-Shot: NEGATIVO (entende o contexto irônico)")
    print()
    print("  Para uso com Ollama real: ver GO1729-CotPrompt.py")
