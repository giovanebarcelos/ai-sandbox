# GO1733-SimulateTool
import ollama
import re

def simulate_tool(action):
    """
    Simula ferramentas (Wikipedia, Calculator)
    """
    # Extrair ferramenta e argumento
    match = re.match(r'(\w+)\[(.*)\]', action.strip())
    if not match:
        return "Erro: formato inválido. Use FERRAMENTA[argumento]"

    tool, arg = match.groups()

    if tool == "BUSCAR":
        # Simular busca (em produção, usaria Wikipedia API)
        knowledge_base = {
            "população de tóquio": "Tóquio tem cerca de 14 milhões de habitantes (2024)",
            "capital da frança": "Paris é a capital da França desde 987 d.C.",
            "quem inventou a internet": "A internet foi desenvolvida pela ARPA em 1969"
        }
        query_lower = arg.lower()
        for key in knowledge_base:
            if key in query_lower:
                return knowledge_base[key]
        return "Informação não encontrada"

    elif tool == "CALCULAR":
        try:
            result = eval(arg)  # ⚠️ CUIDADO: eval é perigoso em produção!
            return f"Resultado: {result}"
        except Exception as e:
            return f"Erro no cálculo: {e}"

    elif tool == "FINALIZAR":
        return arg

    return "Ferramenta desconhecida"

def react_agent(question, max_iterations=5):
    """
    Agente ReAct completo
    """
    prompt = f"""
Você é um assistente que pode PENSAR e AGIR.

Ferramentas:
- BUSCAR[query]: Busca informação
- CALCULAR[expressão]: Calcula
- FINALIZAR[resposta]: Retorna resposta

Formato:
Thought: [raciocínio]
Action: [FERRAMENTA[args]]

Question: {question}

Begin!
Thought:
"""

    conversation = prompt

    for iteration in range(max_iterations):
        # LLM pensa e propõe ação
        response = ollama.chat(
            model='llama3.2',
            messages=[{'role': 'user', 'content': conversation}]
        )

        agent_response = response['message']['content']
        print(f"\n{'='*70}")
        print(f"ITERAÇÃO {iteration + 1}")
        print(f"{'='*70}")
        print(agent_response)

        # Detectar ação
        action_match = re.search(r'Action:\s*(.+)', agent_response)

        if not action_match:
            print("\n✅ Agente finalizou (sem mais ações)")
            break

        action = action_match.group(1).strip()

        # Executar ferramenta
        observation = simulate_tool(action)

        print(f"\n📊 OBSERVATION: {observation}")

        # Atualizar conversação
        conversation += f"\n{agent_response}\nObservation: {observation}\nThought:"

        # Se FINALIZAR, terminar
        if action.startswith("FINALIZAR"):
            print(f"\n🎯 RESPOSTA FINAL: {observation}")
            return observation

    return "Máximo de iterações atingido"

# ════════════════════════════════════════════════════════
# TESTAR ReAct
# ════════════════════════════════════════════════════════

question = "Qual a população de Tóquio multiplicada por 2?"

print("🤖 INICIANDO AGENTE ReAct")
print(f"❓ PERGUNTA: {question}\n")

answer = react_agent(question, max_iterations=5)

print("\n" + "="*70)
print("✅ EXECUÇÃO COMPLETA")
print("="*70)
