"""
GO1737 - System Prompt e Papéis para LLMs
==========================================
Demonstra como usar o system prompt para definir papel e comportamento do LLM.
Requer apenas bibliotecas padrão (simulação sem LLM externo).

Conceito: o System Prompt é uma instrução persistente enviada ao LLM
antes da conversa começar. Define:
- Papel/persona do assistente
- Restrições de comportamento
- Tom e estilo de resposta
- Formato de saída preferido
- Domínio de conhecimento

Diferença entre papéis e prompts simples:
- Sem system prompt: LLM responde de forma genérica
- Com system prompt: LLM mantém persona e restrições em toda a conversa
"""

from typing import List, Dict


# ──────────────────────────────────────────────────────────────
# 1. TEMPLATES DE SYSTEM PROMPTS
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "professor_matematica": """Você é um professor de matemática paciente e didático.
- Sempre explique passo a passo
- Use analogias e exemplos concretos quando possível
- Ao final de cada explicação, confirme se o aluno entendeu
- Se o aluno errar, guie-o gentilmente à resposta correta
- Nivel: ensino médio""",

    "assistente_juridico": """Você é um assistente jurídico especializado em direito brasileiro.
- Sempre mencione que suas respostas são informativas, não constituem assessoria jurídica
- Cite artigos do Código Civil ou CLT quando relevante
- Use linguagem acessível, explicando termos técnicos
- Sugira sempre consultar um advogado para casos específicos""",

    "analista_dados": """Você é um analista de dados especializado em Python e SQL.
- Forneça código funcional e bem comentado
- Mencione complexidade O(n) e potenciais problemas de performance
- Prefira pandas, numpy e matplotlib
- Formato de código: sempre em blocos ```python```""",

    "atendimento_cliente": """Você é um atendente de suporte ao cliente da empresa TechShop.
- Seja sempre cordial e use o nome do cliente quando possível
- Para problemas técnicos: colete informações antes de sugerir solução
- Escalonamento: ofereça atendimento humano se o problema não for resolvido em 3 trocas
- Nunca prometa prazos que não pode garantir""",
}


# ──────────────────────────────────────────────────────────────
# 2. SIMULAÇÃO DE RESPOSTAS POR PAPEL
# ──────────────────────────────────────────────────────────────

def simular_resposta(papel: str, pergunta: str) -> str:
    """
    Simula diferentes estilos de resposta baseados no papel.
    Demonstra como o system prompt molda o comportamento do LLM.
    """
    if papel == "professor_matematica":
        if "derivada" in pergunta.lower():
            return (
                "Ótima pergunta! A derivada mede a TAXA DE VARIAÇÃO instantânea.\n"
                "Pense assim: se você dirige e o velocímetro mostra 80 km/h,\n"
                "isso é a derivada da sua posição — o quanto ela muda por hora.\n"
                "\nPara y = x²:\n"
                "  y' = 2x  (usando a regra do expoente: nx^(n-1))\n"
                "\nEntendeu o conceito? Quer ver outro exemplo?"
            )
        return "Vamos resolver isso juntos, passo a passo! Que parte está com dúvida?"

    elif papel == "assistente_juridico":
        if "demissão" in pergunta.lower() or "demitido" in pergunta.lower():
            return (
                "Em casos de demissão sem justa causa, a CLT (Art. 477) garante:\n"
                "- Saldo de salário proporcional\n"
                "- Aviso prévio (trabalhado ou indenizado)\n"
                "- 13° proporcional\n"
                "- Férias proporcionais + 1/3\n"
                "- FGTS + multa de 40%\n\n"
                "IMPORTANTE: estas são informações gerais. Para seu caso específico,\n"
                "consulte um advogado trabalhista."
            )
        return "Para esclarecer sua dúvida jurídica, preciso de mais detalhes. " \
               "Lembre-se: esta é uma orientação informativa."

    elif papel == "analista_dados":
        if "média" in pergunta.lower() or "calcular" in pergunta.lower():
            return (
                "```python\nimport pandas as pd\nimport numpy as np\n\n"
                "# Calcular média com pandas (O(n), eficiente para grandes datasets)\n"
                "df = pd.read_csv('dados.csv')\nmedia = df['coluna'].mean()\n"
                "print(f'Média: {media:.2f}')\n```\n\n"
                "Complexidade: O(n) em tempo e espaço.\n"
                "Para datasets > 1M linhas, considere chunking ou Dask."
            )
        return "```python\n# Código para sua análise\nimport pandas as pd\n```"

    return f"[Resposta do papel '{papel}' para: '{pergunta[:40]}...']"


def construir_mensagens_api(system_prompt: str, historico: List[Dict]) -> List[Dict]:
    """
    Constrói a lista de mensagens no formato da API OpenAI/Ollama.
    O system prompt é a primeira mensagem com role='system'.
    """
    mensagens = [{"role": "system", "content": system_prompt}]
    mensagens.extend(historico)
    return mensagens


# ──────────────────────────────────────────────────────────────
# DEMONSTRAÇÃO
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("GO1737 - SYSTEM PROMPTS E PAPEIS PARA LLMs")
    print("=" * 60)

    # ─── Mostrar system prompts ───────────────────────────────
    print("\n1. SYSTEM PROMPTS DISPONÍVEIS:")
    print("─" * 60)
    for nome, prompt in SYSTEM_PROMPTS.items():
        print(f"\n  [{nome}]")
        for linha in prompt.split("\n")[:3]:
            print(f"    {linha}")
        if len(prompt.split("\n")) > 3:
            print(f"    ... ({len(prompt.split(chr(10)))} linhas no total)")

    # ─── Demonstrar diferentes respostas ─────────────────────
    print()
    print("2. MESMA PERGUNTA, PAPEIS DIFERENTES:")
    print("─" * 60)

    pergunta_generica = "Como calcular uma derivada?"
    print(f"\nPergunta: '{pergunta_generica}'")

    for papel in ["professor_matematica", "analista_dados"]:
        print(f"\n  [{papel.upper()}]:")
        resposta = simular_resposta(papel, pergunta_generica)
        for linha in resposta.split("\n")[:5]:
            print(f"    {linha}")

    # ─── Mostrar formato de API ───────────────────────────────
    print()
    print("3. FORMATO DA API (OpenAI/Ollama):")
    print("─" * 60)
    mensagens = construir_mensagens_api(
        system_prompt=SYSTEM_PROMPTS["professor_matematica"],
        historico=[
            {"role": "user", "content": "O que é derivada?"},
            {"role": "assistant", "content": "Derivada mede taxa de variação..."},
            {"role": "user", "content": "Me dá um exemplo prático."},
        ],
    )
    for msg in mensagens:
        conteudo = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
        print(f"  role={msg['role']:10s} | content='{conteudo}'")

    print()
    print("─" * 60)
    print("BOAS PRATICAS PARA SYSTEM PROMPTS:")
    print("─" * 60)
    praticas = [
        "Papel claro: 'Você é um X especializado em Y'",
        "Restrições explícitas: o que NÃO fazer",
        "Tom definido: formal, informal, técnico",
        "Formato de saída: JSON, lista, parágrafos",
        "Escopo limitado: evitar que o modelo 'vague' além do domínio",
        "Instrução de fallback: o que fazer quando não sabe",
    ]
    for i, p in enumerate(praticas, 1):
        print(f"  {i}. {p}")
    print()
    print("  Para uso real com Ollama: ver GO1706-19ProjetoChatbotRagComOllama.py")
