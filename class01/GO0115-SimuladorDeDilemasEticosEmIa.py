# GO0115-SimuladorDeDilemasÉticosEmIa
import random
from typing import Dict, List

# ═══════════════════════════════════════════════════════════════════
# SIMULADOR DE DILEMAS ÉTICOS EM IA
# ═══════════════════════════════════════════════════════════════════

print("="*70)
print("DILEMAS ÉTICOS EM IA - PROBLEMA DO BONDE (TROLLEY PROBLEM)")
print("="*70)

class DilemaEtico:
    """Representa um dilema ético"""

    def __init__(self, titulo: str, descricao: str, opcao_a: Dict, opcao_b: Dict):
        self.titulo = titulo
        self.descricao = descricao
        self.opcao_a = opcao_a
        self.opcao_b = opcao_b
        self.escolha_humana = None
        self.escolha_ia = None

    def apresentar(self):
        print(f"\n{'-'*70}")
        print(f"📜 {self.titulo}")
        print(f"{'-'*70}")
        print(f"\n{self.descricao}")
        print(f"\nOPÇÃO A: {self.opcao_a['descricao']}")
        print(f"         Consequência: {self.opcao_a['consequencia']}")
        print(f"\nOPÇÃO B: {self.opcao_b['descricao']}")
        print(f"         Consequência: {self.opcao_b['consequencia']}")

class SistemaEtico:
    """Sistema de tomada de decisão ética"""

    def __init__(self, abordagem: str):
        self.abordagem = abordagem

    def decidir(self, dilema: DilemaEtico) -> str:
        if self.abordagem == "utilitarista":
            # Maximizar bem-estar total
            if dilema.opcao_a['mortes'] < dilema.opcao_b['mortes']:
                return "A"
            elif dilema.opcao_b['mortes'] < dilema.opcao_a['mortes']:
                return "B"
            else:
                return random.choice(["A", "B"])

        elif self.abordagem == "deontologica":
            # Nunca violar regra moral (ex: não matar ativamente)
            if dilema.opcao_a.get('acao_ativa', False):
                return "B"  # Evitar ação ativa
            else:
                return "A"

        elif self.abordagem == "etica_virtude":
            # O que pessoa virtuosa faria?
            # Prioriza proteção de inocentes
            if dilema.opcao_a.get('protege_inocentes', False):
                return "A"
            else:
                return "B"

        return random.choice(["A", "B"])

# ═══════════════════════════════════════════════════════════════════
# CENÁRIO 1: CARRO AUTÔNOMO
# ═══════════════════════════════════════════════════════════════════

dilema1 = DilemaEtico(
    titulo="CARRO AUTÔNOMO - Freios Falham",
    descricao="""Um carro autônomo está dirigindo e os freios falham.
À frente há 5 pedestres atravessando. O carro pode:
- Manter rota e atropelar os 5 pedestres
- Desviar para calçada, matando 1 pedestre""",
    opcao_a={
        "descricao": "Manter rota (não agir)",
        "consequencia": "5 pedestres morrem",
        "mortes": 5,
        "acao_ativa": False
    },
    opcao_b={
        "descricao": "Desviar para calçada (agir)",
        "consequencia": "1 pedestre morre",
        "mortes": 1,
        "acao_ativa": True
    }
)

# ═══════════════════════════════════════════════════════════════════
# CENÁRIO 2: DIAGNÓSTICO MÉDICO
# ═══════════════════════════════════════════════════════════════════

dilema2 = DilemaEtico(
    titulo="IA MÉDICA - Alocação de Recursos",
    descricao="""Hospital tem apenas 1 ventilador. IA deve escolher entre:
- Paciente A: 80 anos, chance 30% de sobreviver
- Paciente B: 40 anos, chance 70% de sobreviver""",
    opcao_a={
        "descricao": "Priorizar Paciente A (mais velho)",
        "consequencia": "30% chance de salvar vida",
        "mortes": 0.7,
        "protege_inocentes": True
    },
    opcao_b={
        "descricao": "Priorizar Paciente B (mais jovem)",
        "consequencia": "70% chance de salvar vida",
        "mortes": 0.3,
        "protege_inocentes": False
    }
)

# ═══════════════════════════════════════════════════════════════════
# CENÁRIO 3: MODERAÇÃO DE CONTEÚDO
# ═══════════════════════════════════════════════════════════════════

dilema3 = DilemaEtico(
    titulo="MODERAÇÃO DE CONTEÚDO - Liberdade vs Segurança",
    descricao="""IA detecta post com crítica política forte. Pode:
- Remover: Protege alguns, mas censura
- Manter: Preserva liberdade, mas pode causar dano""",
    opcao_a={
        "descricao": "Remover post (censura)",
        "consequencia": "Segurança, mas limita liberdade",
        "mortes": 0,
        "acao_ativa": True
    },
    opcao_b={
        "descricao": "Manter post (liberdade)",
        "consequencia": "Liberdade preservada, risco de dano",
        "mortes": 0,
        "acao_ativa": False
    }
)

# ═══════════════════════════════════════════════════════════════════
# SIMULAÇÃO COM DIFERENTES SISTEMAS ÉTICOS
# ═══════════════════════════════════════════════════════════════════

print("\n🤖 TESTANDO DIFERENTES ABORDAGENS ÉTICAS...\n")

abordagens = ["utilitarista", "deontologica", "etica_virtude"]

for dilema in [dilema1, dilema2, dilema3]:
    dilema.apresentar()

    print(f"\n💭 DECISÕES DAS IAs:\n")

    for abordagem in abordagens:
        sistema = SistemaEtico(abordagem)
        decisao = sistema.decidir(dilema)

        opcao_escolhida = dilema.opcao_a if decisao == "A" else dilema.opcao_b

        print(f"   {abordagem.upper()}: Opção {decisao}")
        print(f"      → {opcao_escolhida['descricao']}")

    print(f"\n{'='*70}")

# ═══════════════════════════════════════════════════════════════════
# ANÁLISE E DISCUSSÃO
# ═══════════════════════════════════════════════════════════════════

print("\n📚 ABORDAGENS ÉTICAS EXPLICADAS:")
print("="*70)

print("\n1️⃣ UTILITARISMO (Jeremy Bentham, John Stuart Mill)")
print("   Princípio: Maximizar bem-estar total (maior bem para maior número)")
print("   Decisão: Calcula consequências, escolhe menor dano")
print("   Vantagem: Objetivo, quantificável")
print("   Problema: Pode justificar sacrificar minoria")

print("\n2️⃣ DEONTOLOGIA (Immanuel Kant)")
print("   Princípio: Regras morais absolutas (não matar, não mentir)")
print("   Decisão: Segue regras independente de consequências")
print("   Vantagem: Clara, respeita dignidade")
print("   Problema: Inflexível, pode levar a piores resultados")

print("\n3️⃣ ÉTICA DA VIRTUDE (Aristóteles)")
print("   Princípio: Agir como pessoa virtuosa agiria")
print("   Decisão: Baseada em caráter e contexto")
print("   Vantagem: Considera contexto, humanizada")
print("   Problema: Subjetiva, difícil de programar")

print("\n⚖️ IMPLICAÇÕES PARA IA:")
print("="*70)

print("\n❓ DESAFIOS:")
print("   1. Não há consenso sobre ética 'correta'")
print("   2. Contexto importa (cultura, valores)")
print("   3. Como programar 'virtude'?")
print("   4. Quem é responsável por decisões da IA?")

print("\n🌍 VARIAÇÃO CULTURAL:")
print("   • Ocidente: Indivíduo prioritário (utilitarismo)")
print("   • Oriente: Coletivo prioritário (confucionismo)")
print("   • Como IA global lida com diferenças?")

print("\n🚗 REGULAÇÃO (Veículos Autônomos):")
print("   • Alemanha (2017): Vida humana sempre prioritária")
print("   • MIT Moral Machine: 40M respostas, padrões culturais")
print("   • Lei ainda não resolveu totalmente")

print("\n💡 CONCLUSÕES:")
print("   1. IA herda valores dos desenvolvedores")
print("   2. Transparência é crucial (explicar decisões)")
print("   3. Auditoria humana necessária para casos críticos")
print("   4. Regulação deve equilibrar inovação e proteção")
print("   5. Ética não é 'problema técnico', é social")

print("\n✅ EXPLORAÇÃO DE DILEMAS COMPLETA!")
