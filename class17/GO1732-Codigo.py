"""
GO1732 - Template ReAct (Reason + Act)
=======================================
Demonstra o template de prompt para agentes ReAct.
Requer apenas bibliotecas padrão.

Conceito: ReAct (Yao et al. 2022) combina raciocínio (Thought) e
ação (Action) em um loop iterativo:

  Thought: Preciso buscar X
  Action: BUSCAR[X]
  Observation: resultado do BUSCAR
  Thought: Agora sei Y
  Action: CALCULAR[Y * 2]
  Observation: resultado do CALCULAR
  ...
  Answer: resposta final

Diferença vs. CoT (Chain-of-Thought):
- CoT: raciocínio interno apenas (sem ações externas)
- ReAct: raciocínio + chamadas a ferramentas externas (APIs, busca, etc.)

Ver GO1733 para implementação completa com Ollama.
"""

from typing import List, Tuple


# ──────────────────────────────────────────────────────────────
# 1. TEMPLATE REACT
# ──────────────────────────────────────────────────────────────

REACT_TEMPLATE = """Você é um assistente que pode PENSAR e AGIR.

Ferramentas disponíveis:
- BUSCAR[query]: Busca informação na Wikipedia
- CALCULAR[expressão]: Calcula expressões matemáticas
- FINALIZAR[resposta]: Retorna resposta final

Use o formato:
Thought: [seu raciocínio]
Action: [FERRAMENTA[argumentos]]
Observation: [resultado da ação]
... (repita Thought/Action/Observation até ter resposta)
Thought: Agora sei a resposta
Answer: [resposta final]

Exemplo:
Question: Qual a população de Tóquio?

Thought: Preciso buscar informação sobre Tóquio
Action: BUSCAR[População de Tóquio]
Observation: Tóquio tem aproximadamente 14 milhões de habitantes
Thought: Agora sei a resposta
Answer: Tóquio tem aproximadamente 14 milhões de habitantes

Agora responda:
Question: {question}
"""


def construir_prompt_react(questao: str) -> str:
    return REACT_TEMPLATE.format(question=questao)


# ──────────────────────────────────────────────────────────────
# 2. SIMULAÇÃO DE UM TRACE REACT
# ──────────────────────────────────────────────────────────────

def simular_trace_react(questao: str) -> List[Tuple[str, str]]:
    """
    Simula um trace de execução ReAct para demonstração.
    Retorna lista de (tipo, conteúdo) representando cada etapa.
    """
    # Trace simulado para a questão "população de Tóquio × 2"
    if "tóquio" in questao.lower() or "toquio" in questao.lower():
        return [
            ("Thought", "Primeiro preciso descobrir a população de Tóquio."),
            ("Action", "BUSCAR[população de Tóquio]"),
            ("Observation", "Tóquio tem aproximadamente 14 milhões de habitantes (2024)."),
            ("Thought", "Agora preciso multiplicar por 2."),
            ("Action", "CALCULAR[14000000 * 2]"),
            ("Observation", "Resultado: 28000000"),
            ("Thought", "Tenho a resposta final."),
            ("Answer", "O dobro da população de Tóquio é aproximadamente 28 milhões de pessoas."),
        ]
    elif "capital" in questao.lower() and "brasil" in questao.lower():
        return [
            ("Thought", "Preciso saber qual é a capital do Brasil."),
            ("Action", "BUSCAR[capital do Brasil]"),
            ("Observation", "A capital do Brasil é Brasília, inaugurada em 1960."),
            ("Thought", "Tenho a resposta."),
            ("Answer", "A capital do Brasil é Brasília."),
        ]
    else:
        return [
            ("Thought", "Vou buscar informações sobre a questão."),
            ("Action", f"BUSCAR[{questao[:30]}]"),
            ("Observation", "Informação encontrada na base de conhecimento."),
            ("Thought", "Posso responder agora."),
            ("Answer", "Resposta baseada na busca realizada."),
        ]


# ──────────────────────────────────────────────────────────────
# DEMONSTRAÇÃO
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("GO1732 - TEMPLATE REACT (Reason + Act)")
    print("=" * 60)

    # ─── Mostrar o template ──────────────────────────────────
    questao = "Qual a população de Tóquio multiplicada por 2?"
    print("\nTEMPLATE DO PROMPT REACT:")
    print("─" * 60)
    print(construir_prompt_react(questao))

    # ─── Simular trace de execução ────────────────────────────
    print("─" * 60)
    print("TRACE DE EXECUÇÃO SIMULADO:")
    print("─" * 60)
    trace = simular_trace_react(questao)
    for tipo, conteudo in trace:
        prefixo = {
            "Thought": "🤔 Thought    ",
            "Action": "⚡ Action     ",
            "Observation": "👁  Observation",
            "Answer": "✅ Answer     ",
        }.get(tipo, tipo)
        print(f"\n  {prefixo}: {conteudo}")

    # ─── Comparação ReAct vs CoT ──────────────────────────────
    print()
    print("─" * 60)
    print("COMPARAÇÃO: CoT vs ReAct")
    print("─" * 60)
    print()
    print("  CoT (Chain-of-Thought):")
    print("    Thought: 'Tóquio tem ~14M hab. 14M × 2 = 28M.'")
    print("    Problema: modelo pode ALUCINAR o valor da população!")
    print()
    print("  ReAct:")
    print("    Thought → BUSCAR[Tóquio] → Observation: '14M (fonte real)'")
    print("    Thought → CALCULAR[14M × 2] → Observation: '28M'")
    print("    Vantagem: usa ferramentas REAIS, não memória potencialmente errada.")
    print()
    print("─" * 60)
    print("FERRAMENTAS TÍPICAS DE UM AGENTE REACT:")
    print("─" * 60)
    ferramentas = [
        ("BUSCAR", "Wikipedia, Google, base interna"),
        ("CALCULAR", "Expressões matemáticas via Python eval()"),
        ("SQL", "Consultar banco de dados"),
        ("API", "Previsão do tempo, preços, etc."),
        ("CODIGO", "Executar código Python"),
        ("RAG", "Buscar em documentos internos"),
    ]
    for nome, desc in ferramentas:
        print(f"  {nome:10s}: {desc}")

    print()
    print("  Ver GO1733 para implementação completa com Ollama.")
