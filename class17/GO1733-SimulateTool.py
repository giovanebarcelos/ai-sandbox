"""
GO1733 - Agente ReAct com Ferramentas (Tool-Calling)
=====================================================
Demonstra um agente ReAct (Reasoning + Acting) que usa ferramentas
para responder perguntas que requerem múltiplos passos.

Conceito:
  ReAct = Reason + Act
  O agente alterna entre:
    Thought: raciocínio sobre o problema
    Action:  execução de uma ferramenta
    Observation: resultado da ferramenta
    ...repete até...
    Answer: resposta final

Ferramentas disponíveis:
  BUSCAR[query]      — busca em base de conhecimento
  CALCULAR[expr]     — calcula expressão matemática
  FINALIZAR[resposta]— retorna resposta ao usuário

Instalação (opcional — só para usar Ollama real):
  pip install ollama
  ollama pull llama3.2
"""

import re
from typing import Optional


# ─────────────────────────────────────────────────────────────────
# SIMULAÇÃO DAS FERRAMENTAS
# ─────────────────────────────────────────────────────────────────

# Base de conhecimento simulada
KNOWLEDGE_BASE = {
    "população de tóquio": "Tóquio tem cerca de 14 milhões de habitantes (2024)",
    "capital da frança": "Paris é a capital da França desde 987 d.C.",
    "quem inventou a internet": "A internet foi desenvolvida pela ARPA em 1969",
    "distância terra lua": "A distância média Terra-Lua é de 384.400 km",
    "velocidade da luz": "A velocidade da luz no vácuo é 299.792 km/s",
    "pib do brasil": "O PIB do Brasil em 2023 foi de aproximadamente R$ 10,9 trilhões",
    "maior planeta": "Júpiter é o maior planeta do sistema solar",
    "temperatura de ebulição da água": "A água ferve a 100°C ao nível do mar",
}


def simulate_tool(action: str) -> str:
    """
    Simula a execução de ferramentas externas.
    Em produção: BUSCAR conectaria à Wikipedia/API real;
                 CALCULAR usaria um interpretador seguro.
    """
    # Extrair ferramenta e argumento do formato FERRAMENTA[argumento]
    match = re.match(r'(\w+)\[(.*)\]', action.strip())
    if not match:
        return "Erro: formato inválido. Use FERRAMENTA[argumento]"

    tool, arg = match.groups()
    tool = tool.upper()

    if tool == "BUSCAR":
        # Simular busca na base de conhecimento
        query_lower = arg.lower()
        for key in KNOWLEDGE_BASE:
            if key in query_lower or any(w in query_lower for w in key.split()):
                return KNOWLEDGE_BASE[key]
        return f"Informação sobre '{arg}' não encontrada na base local."

    elif tool == "CALCULAR":
        try:
            # AVISO: eval é perigoso em produção! Use um parser seguro.
            # Aqui é didático; em produção use numexpr ou sympy.
            resultado = eval(arg, {"__builtins__": {}}, {})
            return f"Resultado: {resultado}"
        except Exception as e:
            return f"Erro no cálculo: {e}"

    elif tool == "FINALIZAR":
        return arg

    return f"Ferramenta desconhecida: {tool}. Use BUSCAR, CALCULAR ou FINALIZAR."


# ─────────────────────────────────────────────────────────────────
# AGENTE REACT COM OLLAMA (COM FALLBACK)
# ─────────────────────────────────────────────────────────────────

def react_agent_ollama(question: str, max_iterations: int = 5) -> str:
    """
    Agente ReAct usando Ollama como LLM de raciocínio.
    Requer: pip install ollama && ollama pull llama3.2
    """
    import ollama

    prompt = f"""Você é um assistente que pode PENSAR e AGIR.

Ferramentas:
- BUSCAR[query]: Busca informação
- CALCULAR[expressão]: Calcula expressão matemática
- FINALIZAR[resposta]: Retorna resposta final

Formato OBRIGATÓRIO (uma linha cada):
Thought: [raciocínio sobre o próximo passo]
Action: FERRAMENTA[argumento]

Quando tiver a resposta, use:
Action: FINALIZAR[resposta completa]

Pergunta: {question}

Thought:"""

    conversation = prompt

    for iteration in range(max_iterations):
        response = ollama.chat(
            model='llama3.2',
            messages=[{'role': 'user', 'content': conversation}]
        )
        agent_response = response['message']['content']

        print(f"\n{'─'*60}")
        print(f"ITERACAO {iteration + 1}:")
        print(f"{'─'*60}")
        print(agent_response)

        # Detectar ação na resposta
        action_match = re.search(r'Action:\s*(.+)', agent_response)

        if not action_match:
            print("\nAgente finalizou sem acao explícita.")
            break

        action = action_match.group(1).strip()
        observation = simulate_tool(action)

        print(f"\nFERRAMENTA EXECUTADA: {action}")
        print(f"RESULTADO: {observation}")

        # Adicionar observação para próxima iteração
        conversation += f"\n{agent_response}\nObservation: {observation}\nThought:"

        # Se FINALIZAR foi chamado, terminar
        if action.upper().startswith("FINALIZAR"):
            print(f"\nRESPOSTA FINAL: {observation}")
            return observation

    return "Maximo de iteracoes atingido sem resposta final."


# ─────────────────────────────────────────────────────────────────
# DEMO SEM OLLAMA — TRACE SIMULADO
# ─────────────────────────────────────────────────────────────────

def demo_react_simulado(question: str) -> str:
    """
    Simula um trace ReAct sem usar LLM real.
    Demonstra o padrão Thought→Action→Observation→Answer.
    """
    print(f"\nPERGUNTA: {question}")
    print("─" * 60)

    # Trace hardcoded que demonstra o padrão ReAct
    if "multiplicada" in question.lower() or "vezes" in question.lower():
        steps = [
            ("Thought: Preciso encontrar a população de Tóquio primeiro.",
             "Action: BUSCAR[população de tóquio]"),
            ("Thought: Agora preciso multiplicar 14.000.000 por 2.",
             "Action: CALCULAR[14000000 * 2]"),
            ("Thought: Tenho o resultado. Posso finalizar.",
             "Action: FINALIZAR[A população de Tóquio (~14 milhões) multiplicada por 2 é 28.000.000.]"),
        ]
    else:
        steps = [
            ("Thought: Preciso buscar essa informação na base de conhecimento.",
             f"Action: BUSCAR[{question}]"),
            ("Thought: Encontrei a informação. Posso finalizar.",
             "Action: FINALIZAR[Informação recuperada com sucesso via busca.]"),
        ]

    answer = ""
    for i, (thought, action) in enumerate(steps, 1):
        print(f"\n[ITERACAO {i}]")
        print(thought)
        print(action)

        # Executar ferramenta
        action_str = action.replace("Action: ", "")
        observation = simulate_tool(action_str)
        print(f"Observation: {observation}")

        if action_str.upper().startswith("FINALIZAR"):
            answer = observation
            break

    print(f"\n{'='*60}")
    print(f"RESPOSTA FINAL: {answer}")
    return answer


if __name__ == "__main__":
    print("=" * 60)
    print("GO1733 - AGENTE REACT COM FERRAMENTAS")
    print("=" * 60)

    print("\nCONCEITO REACT:")
    print()
    print("  Thought: [raciocínio]")
    print("  Action:  FERRAMENTA[argumento]")
    print("  Observation: [resultado da ferramenta]")
    print("  ... repetir até ...")
    print("  Action:  FINALIZAR[resposta completa]")

    print()
    print("─" * 60)
    print("DEMO SEM OLLAMA — TRACE SIMULADO:")
    print("─" * 60)

    pergunta = "Qual a população de Tóquio multiplicada por 2?"
    demo_react_simulado(pergunta)

    print()
    print("─" * 60)
    print("TESTANDO FERRAMENTAS DIRETAMENTE:")
    print("─" * 60)
    testes = [
        "BUSCAR[população de tóquio]",
        "BUSCAR[capital da frança]",
        "CALCULAR[14000000 * 2]",
        "CALCULAR[384400 / 299792]",
        "FINALIZAR[Resposta: 28 milhões de pessoas]",
    ]
    for t in testes:
        resultado = simulate_tool(t)
        print(f"\n  Acao: {t}")
        print(f"  Resultado: {resultado}")

    print()
    print("─" * 60)
    print("PARA USAR COM OLLAMA REAL:")
    print("─" * 60)
    print("  pip install ollama")
    print("  ollama pull llama3.2")
    print("  Execute: react_agent_ollama('Qual a população de Tóquio × 2?')")

    try:
        import ollama
        print("\nOllama disponível! Tentando executar agente real...")
        print("─" * 60)
        react_agent_ollama(pergunta)
    except ImportError:
        print("\nOllama nao instalado — demo simulado acima e o equivalente.")
    except Exception as e:
        print(f"\nErro ao conectar ao Ollama: {e}")
        print("Certifique-se que 'ollama serve' esta rodando.")
