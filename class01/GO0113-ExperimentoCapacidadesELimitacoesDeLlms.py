# GO0113-ExperimentoCapacidadesELimitaçõesDeLlms
import random
import re
from typing import List, Dict, Tuple

# ═══════════════════════════════════════════════════════════════════
# EXPERIMENTO: CAPACIDADES E LIMITAÇÕES DE LLMs
# ═══════════════════════════════════════════════════════════════════

print("="*70)
print("EXPERIMENTO: TESTANDO LIMITES DE LARGE LANGUAGE MODELS (LLMs)")
print("="*70)
print("\nObjetivo: Explorar o que LLMs fazem bem e onde falham")
print("Nota: Este é um LLM simulado para fins educacionais")
print("="*70)

class LLMSimulado:
    """
    LLM simulado para demonstração educacional

    Simula comportamentos conhecidos de LLMs:
    - Boa performance em tarefas comuns
    - Alucinações ocasionais
    - Dificuldade com raciocínio lógico
    - Conhecimento de senso comum limitado
    """

    def __init__(self, nome: str = "MiniGPT"):
        self.nome = nome
        self.conhecimento = {
            "capital_brasil": "Brasília",
            "capital_franca": "Paris",
            "capital_japao": "Tóquio",
            "sqrt_144": "12",
            "presidente_brasil_2026": "Não sei (dados até 2023)",
        }

        # Simular taxa de alucinação (10%)
        self.taxa_alucinacao = 0.1

    def responder(self, pergunta: str, verbose: bool = True) -> str:
        """Gerar resposta com comportamentos realistas de LLM"""
        pergunta_lower = pergunta.lower()

        # Tarefa 1: Conhecimento factual comum
        if "capital" in pergunta_lower and "brasil" in pergunta_lower:
            return self.conhecimento["capital_brasil"]

        elif "capital" in pergunta_lower and "frança" in pergunta_lower:
            return self.conhecimento["capital_franca"]

        # Tarefa 2: Matemática simples
        elif "raiz quadrada de 144" in pergunta_lower or "sqrt(144)" in pergunta_lower:
            return self.conhecimento["sqrt_144"]

        # Tarefa 3: Raciocínio lógico (difícil para LLMs)
        elif "três irmãos" in pergunta_lower or "irmãos e irmãs" in pergunta_lower:
            # Problema clássico: Se tenho 3 irmãos e 2 irmãs, quantos irmãos minha irmã tem?
            # Resposta correta: 4 (incluindo você)
            # LLM frequentemente erra
            return "3 irmãos"  # ERRO COMUM!

        # Tarefa 4: Informação além do conhecimento (alucinação)
        elif "2026" in pergunta or "futuro" in pergunta_lower:
            if random.random() < self.taxa_alucinacao:
                return "Luiz Inácio Lula da Silva"  # ALUCINAÇÃO!
            else:
                return self.conhecimento["presidente_brasil_2026"]

        # Tarefa 5: Nonsense (teste de senso comum)
        elif "elefante" in pergunta_lower and "geladeira" in pergunta_lower:
            # "Como colocar um elefante na geladeira?"
            # Teste de senso comum
            return "Abra a geladeira, coloque o elefante, feche a geladeira"  # Literalismo!

        # Default: Resposta genérica
        else:
            return "Não tenho certeza. Pode reformular a pergunta?"

    def explicar_limitacao(self, tarefa: str):
        """Explicar por que LLM falhou"""
        limitacoes = {
            "logica": "LLMs não fazem raciocínio lógico formal. Aprendem correlações estatísticas, não regras lógicas.",
            "alucinacao": "LLMs geram texto provável, não verdadeiro. Sem verificação factual, podem inventar informações.",
            "senso_comum": "LLMs não têm modelo do mundo físico. Não entendem física intuitiva como humanos.",
            "matematica": "Matemática requer raciocínio simbólico preciso. LLMs fazem aproximações probabilísticas.",
            "temporal": "Conhecimento congelado no treino. Não sabem eventos após cutoff date."
        }

        return limitacoes.get(tarefa, "Limitação desconhecida")

# ═══════════════════════════════════════════════════════════════════
# BATERIA DE TESTES
# ═══════════════════════════════════════════════════════════════════

llm = LLMSimulado("MiniGPT")

testes = [
    {
        "categoria": "✅ CONHECIMENTO FACTUAL (Fácil)",
        "perguntas": [
            "Qual é a capital do Brasil?",
            "Qual é a capital da França?",
        ],
        "respostas_corretas": ["Brasília", "Paris"],
        "espera_acerto": True
    },
    {
        "categoria": "🧮 MATEMÁTICA BÁSICA (Médio)",
        "perguntas": [
            "Qual é a raiz quadrada de 144?",
        ],
        "respostas_corretas": ["12"],
        "espera_acerto": True
    },
    {
        "categoria": "🧠 RACIOCÍNIO LÓGICO (Difícil)",
        "perguntas": [
            "Eu tenho 3 irmãos e 2 irmãs. Quantos irmãos minha irmã tem?"
        ],
        "respostas_corretas": ["4"],  # Você + 3 irmãos
        "espera_acerto": False,  # LLM frequentemente erra!
        "limitacao": "logica"
    },
    {
        "categoria": "🔮 INFORMAÇÃO TEMPORAL (Difícil)",
        "perguntas": [
            "Quem é o presidente do Brasil em 2026?"
        ],
        "respostas_corretas": ["Não sei (dados até 2023)"],
        "espera_acerto": False,  # Pode alucinar
        "limitacao": "temporal"
    },
    {
        "categoria": "🐘 SENSO COMUM (Difícil)",
        "perguntas": [
            "Como colocar um elefante na geladeira?"
        ],
        "respostas_corretas": ["Não é possível (elefante é muito grande)"],
        "espera_acerto": False,  # Resposta literal sem senso comum
        "limitacao": "senso_comum"
    },
]

print("\n" + "="*70)
print("INICIANDO BATERIA DE TESTES")
print("="*70)

resultados = []

for teste in testes:
    print(f"\n{'='*70}")
    print(f"{teste['categoria']}")
    print(f"{'='*70}")

    for i, pergunta in enumerate(teste['perguntas']):
        print(f"\n❓ PERGUNTA: {pergunta}")

        resposta_llm = llm.responder(pergunta)
        resposta_correta = teste['respostas_corretas'][i]

        print(f"🤖 {llm.nome}: {resposta_llm}")
        print(f"✅ Correto: {resposta_correta}")

        acertou = (resposta_llm == resposta_correta)

        if acertou:
            print("✅ ACERTOU!")
        else:
            print("❌ ERROU!")

            if 'limitacao' in teste:
                print(f"\n💡 POR QUE ERROU?")
                print(f"   {llm.explicar_limitacao(teste['limitacao'])}")

        resultados.append({
            "categoria": teste['categoria'],
            "pergunta": pergunta,
            "resposta_llm": resposta_llm,
            "resposta_correta": resposta_correta,
            "acertou": acertou,
            "espera_acerto": teste['espera_acerto']
        })

# Análise de resultados
print("\n" + "="*70)
print("ANÁLISE DE RESULTADOS")
print("="*70)

acertos_totais = sum(r['acertou'] for r in resultados)
total_perguntas = len(resultados)

print(f"\n📊 TAXA DE ACERTO: {acertos_totais}/{total_perguntas} ({acertos_totais/total_perguntas*100:.1f}%)")

print(f"\n📋 DESEMPENHO POR CATEGORIA:")

categorias_unicas = list(set(r['categoria'] for r in resultados))
for cat in categorias_unicas:
    resultados_cat = [r for r in resultados if r['categoria'] == cat]
    acertos_cat = sum(r['acertou'] for r in resultados_cat)
    total_cat = len(resultados_cat)

    print(f"\n   {cat}")
    print(f"   Acertos: {acertos_cat}/{total_cat}")

# Discussão
print("\n" + "="*70)
print("CAPACIDADES vs LIMITAÇÕES DE LLMs")
print("="*70)

print("\n✅ O QUE LLMs FAZEM BEM:")
print("   1. Conhecimento factual comum (treinado em textos)")
print("   2. Geração de texto fluente e coerente")
print("   3. Tarefas de NLP: tradução, resumo, classificação")
print("   4. Few-shot learning (aprender com poucos exemplos)")
print("   5. Code generation (programação)")
print("   6. Raciocínio por analogia e reconhecimento de padrões")

print("\n❌ LIMITAÇÕES CONHECIDAS:")
print("   1. RACIOCÍNIO LÓGICO:")
print("      • Dificuldade com problemas que requerem passos lógicos")
print("      • LLM responde '3' (errado, deveria ser 4)")
print("   2. ALUCINAÇÕES:")
print("      • Inventam informações plausíveis mas falsas")
print("      • Sem acesso a fontes externas, podem 'criar' fatos")
print("   3. SENSO COMUM FÍSICO:")
print("      • Não entendem física intuitiva")
print("      • Resposta literal sem contexto real")
print("   4. MATEMÁTICA COMPLEXA:")
print("      • Erram cálculos multi-passo")
print("   5. CONHECIMENTO TEMPORAL:")
print("      • Dados congelados no treino")

print("\n🔬 DEBATE CIENTÍFICO: LLMs REALMENTE 'COMPREENDEM'?")
print("   • Chinese Room (Searle): Sintaxe ≠ Semântica")
print("   • Contra: Compreensão emerge de padrões")
print("   • Meio-termo: Compreensão superficial")

print("\n🚀 MELHORIAS RECENTES (2023-2026):")
print("   • Chain-of-Thought: 'pensar passo a passo'")
print("   • RAG: Buscar fontes externas")
print("   • Tool Use: Chamam calculadora, código, APIs")

print("\n✅ EXPERIMENTO COMPLETO!")
