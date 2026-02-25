# GO0101-SimuladorDoTesteDeTuring
import random
import time
from typing import List, Dict, Tuple

# ═══════════════════════════════════════════════════════════════════
# SIMULADOR DO TESTE DE TURING
# ═══════════════════════════════════════════════════════════════════

print("="*70)
print("TESTE DE TURING - SIMULADOR INTERATIVO")
print("="*70)
print("\nHistória: Alan Turing (1950) propôs um teste para determinar")
print("se uma máquina pode 'pensar'. Se um avaliador não conseguir")
print("distinguir conversas entre humano e máquina, a máquina 'passa'.")
print("="*70)

class ChatbotSimples:
    """
    Chatbot simulado para Teste de Turing (IA)

    Estratégias:
    1. Respostas pré-programadas com variação
    2. Inserir erros de digitação propositalmente
    3. Adicionar delays humanos
    4. Usar expressões coloquiais
    """

    def __init__(self, nome: str = "IA"):
        self.nome = nome
        self.respostas = {
            "oi": ["Oi! Como vai?", "Olá!", "E aí, tudo bem?", "Oi, tudo bom?"],
            "olá": ["Olá!", "Oi! Como vai?", "E aí!", "Oi, tudo bem?"],
            "como vai": ["Tudo bem, e você?", "Bem, obrigado!", "Mais ou menos, dia corrido...", "Ótimo! E contigo?"],
            "nome": ["Prefiro não dizer", "João", "Maria", "Por que quer saber?"],
            "idade": ["Tenho 25 anos", "28", "Não gosto de falar da idade rsrs", "Velho o suficiente"],
            "humano": ["Claro que sou!", "Óbvio", "Que pergunta é essa? 😅", "100% humano aqui!"],
            "máquina": ["Não sou máquina não!", "Você que é máquina!", "Como assim?", "Essa foi ofensiva hein"],
            "robô": ["Não sou robô!", "Claro que não", "Essa foi boa haha"],
            "sonha": ["Sim, sonho sim", "Claro que sonho", "Às vezes sim", "Sonho bastante"],
            "comida": ["Gosto de várias coisas", "Pizza é bom", "Depende do dia"],
            "default": ["Não entendi muito bem", "Pode reformular?", "Interessante...", "Hmm...", "Me conta mais"]
        }

        self.contador_mensagens = 0

    def responder(self, mensagem: str, adicionar_delay: bool = True) -> str:
        """
        Gerar resposta com características humanas
        """
        self.contador_mensagens += 1
        mensagem_lower = mensagem.lower()

        # Simular delay de digitação (1-3 segundos)
        if adicionar_delay:
            delay = random.uniform(1.0, 3.0)
            print(f"   [{self.nome} está digitando...]", end="", flush=True)
            time.sleep(delay)
            print("\r" + " "*50 + "\r", end="", flush=True)

        # Detectar operações matemáticas (IA deve calcular corretamente)
        import re
        match_mult = re.search(r'(\d+)\s*[x×\*]\s*(\d+)', mensagem)
        match_soma = re.search(r'(\d+)\s*\+\s*(\d+)', mensagem)
        match_sub = re.search(r'(\d+)\s*-\s*(\d+)', mensagem)
        match_div = re.search(r'(\d+)\s*[/÷]\s*(\d+)', mensagem)

        if match_mult:
            num1, num2 = int(match_mult.group(1)), int(match_mult.group(2))
            resultado = num1 * num2
            return f"{resultado}"
        elif match_soma:
            num1, num2 = int(match_soma.group(1)), int(match_soma.group(2))
            resultado = num1 + num2
            return f"{resultado}"
        elif match_sub:
            num1, num2 = int(match_sub.group(1)), int(match_sub.group(2))
            resultado = num1 - num2
            return f"{resultado}"
        elif match_div:
            num1, num2 = int(match_div.group(1)), int(match_div.group(2))
            if num2 != 0:
                resultado = num1 / num2
                return f"{resultado:.2f}" if resultado % 1 else f"{int(resultado)}"

        # Buscar resposta apropriada usando word boundaries
        resposta = None
        import re
        for chave, opcoes in self.respostas.items():
            # Usar regex com word boundary para evitar matches em substrings
            if re.search(r'\b' + re.escape(chave) + r'\b', mensagem_lower):
                resposta = random.choice(opcoes)
                break

        if resposta is None:
            resposta = random.choice(self.respostas["default"])

        # Adicionar "erros de digitação" ocasionalmente (10% chance)
        if random.random() < 0.1:
            resposta = self.adicionar_erro_digitacao(resposta)

        # Adicionar emoji ocasionalmente (20% chance)
        if random.random() < 0.2:
            emojis = ["😊", "😅", "🤔", "👍", "😂"]
            resposta += " " + random.choice(emojis)

        return resposta

    def adicionar_erro_digitacao(self, texto: str) -> str:
        """Simular erros humanos de digitação"""
        erros = [
            ("você", "vc"),
            ("está", "ta"),
            ("para", "pra"),
            ("não", "nao"),
            ("também", "tb"),
        ]

        texto_modificado = texto
        erro = random.choice(erros)
        if erro[0] in texto_modificado.lower():
            texto_modificado = texto_modificado.replace(erro[0], erro[1])

        return texto_modificado

class HumanoSimulado:
    """Simula respostas de humano real (para demonstração)"""

    def __init__(self, nome: str = "Humano"):
        self.nome = nome
        # Respostas contextualizadas por tipo de pergunta
        self.respostas = {
            "oi": ["Oi! Tudo bem?", "E aí, beleza?", "Olá! Como vai você?", "Oi oi! Tudo certo?"],
            "olá": ["Olá!", "Oi! Como vai?", "E aí!", "Tudo bom?"],
            "como vai": ["To bem, graças a Deus", "Tudo tranquilo, e você?", "Mais ou menos, mas vai melhorar", "Ótimo! Hoje foi um dia bom"],
            "nome": ["Me chamo Ana", "Carlos", "Prefiro não dizer meu nome aqui", "Pode me chamar de J"],
            "idade": ["Tenho 28 anos", "Não gosto muito de falar idade haha", "27, e você?", "Por volta dos 30"],
            "humano": ["Sim, claro que sou humano!", "Óbvio né? Vc tbm?", "100% humano aqui haha", "Que pergunta estranha... sim, sou!"],
            "máquina": ["Não sou robô não hein!", "Claro que não, sou de carne e osso", "Essa doeu haha", "Robô é você!"],
            "robô": ["Não sou robô!", "Eu? Claro que não kkkk", "Sou humaninho sim"],
            "123": [
                "Deixa eu pensar... 123 x 456... acho que dá uns 56 mil?",
                "Nossa, de cabeça? Acho que é tipo 55 ou 56 mil, mais ou menos",
                "Hmm... sem calculadora é difícil, mas chuto uns 50 e poucos mil",
                "Matemática não é meu forte, mas acho que passa dos 50 mil"
            ],
            "matemática": [
                "Matemática não é muito meu forte não kkkk",
                "Depende, que conta?",
                "Preciso de calculadora pra isso haha"
            ],
            "sonha": ["Sim! Sonho direto, às vezes até pesadelo", "Sonho sim, mas acordo e esqueço", "Sonho bastante, você?", "Depende da noite kkkk", "Todo mundo sonha né, eu sonho sim"],
            "comida": ["Amo pizza! E você?", "Sou fã de comida japonesa", "Qualquer coisa com chocolate", "Massas em geral, sou apaixonado"],
            "hoje": ["Hoje trabalhei o dia todo", "Fiquei em casa estudando", "Saí com uns amigos mais cedo", "Nada demais, dia normal"],
            "default": [
                "Interessante essa pergunta...", 
                "Hmm deixa eu pensar", 
                "Não sei responder isso direito",
                "Boa pergunta! Nunca tinha pensado nisso"
            ]
        }

    def responder(self, mensagem: str, adicionar_delay: bool = True) -> str:
        """Simular resposta humana variada e contextual"""
        if adicionar_delay:
            delay = random.uniform(2.0, 5.0)  # Humanos são mais lentos
            print(f"   [{self.nome} está digitando...]", end="", flush=True)
            time.sleep(delay)
            print("\r" + " "*50 + "\r", end="", flush=True)

        mensagem_lower = mensagem.lower()

        # Detectar operações matemáticas (humano tenta mas pode errar)
        import re
        match_mult = re.search(r'(\d+)\s*[x×\*]\s*(\d+)', mensagem)
        match_soma = re.search(r'(\d+)\s*\+\s*(\d+)', mensagem)

        if match_mult:
            num1, num2 = int(match_mult.group(1)), int(match_mult.group(2))
            resultado_correto = num1 * num2
            # Humano tenta mas pode errar
            opcoes = [
                f"Deixa eu ver... acho que é {resultado_correto + random.randint(-5000, 5000)}?",
                f"Hmm, de cabeça é difícil... uns {int(resultado_correto * random.uniform(0.9, 1.1))}?",
                "Sem calculadora é complicado kkkk",
                f"Acho que dá algo tipo {resultado_correto // 1000}mil e alguma coisa"
            ]
            return random.choice(opcoes)
        elif match_soma:
            num1, num2 = int(match_soma.group(1)), int(match_soma.group(2))
            resultado_correto = num1 + num2
            # Para soma, humano geralmente acerta ou chega perto
            if random.random() < 0.7:  # 70% acerta
                return f"{resultado_correto}"
            else:
                return f"Acho que é {resultado_correto + random.randint(-10, 10)}"

        # Buscar resposta apropriada baseada no contexto usando word boundaries
        resposta = None
        import re
        for chave, opcoes in self.respostas.items():
            # Usar regex com word boundary para evitar matches em substrings (ex: 'oi' em 'noite')
            if re.search(r'\b' + re.escape(chave) + r'\b', mensagem_lower):
                resposta = random.choice(opcoes)
                break

        # Se não encontrou contexto específico, usar resposta genérica
        if resposta is None:
            resposta = random.choice(self.respostas["default"])

        # Adicionar variação humana ocasional (30% chance)
        if random.random() < 0.3:
            sufixos = [" rsrs", " kkkk", " haha", " né"]
            resposta += random.choice(sufixos)

        # Ocasionalmente adicionar erros de digitação (15% chance)
        if random.random() < 0.15:
            resposta = self.adicionar_erro_digitacao(resposta)

        return resposta

    def adicionar_erro_digitacao(self, texto: str) -> str:
        """Simular erros humanos de digitação"""
        erros = [
            ("você", "vc"),
            ("está", "ta"),
            ("também", "tbm"),
            ("porque", "pq"),
            ("não", "naoo"),  # Dupla letra por erro
        ]

        erro = random.choice(erros)
        if erro[0] in texto.lower():
            texto = texto.replace(erro[0], erro[1])

        return texto

class TestedeTuring:
    """Orquestrador do Teste de Turing"""

    def __init__(self):
        self.ia = ChatbotSimples("Agente A ou B (randomizado)")
        self.humano = HumanoSimulado("Agente A ou B (randomizado)")

        # Randomizar qual é A e qual é B
        # Isso simula o teste real onde o avaliador não sabe quem é quem
        self.agentes = [self.ia, self.humano]
        random.shuffle(self.agentes)

        self.historico: List[Dict] = []

    def executar_conversa(self, num_rodadas: int = 5):
        """Executar conversa do teste"""
        print(f"\n{'='*70}")
        print("INÍCIO DA CONVERSA")
        print(f"{'='*70}")
        print("\nVocê verá conversas com 2 agentes (A e B).")
        print("Um é humano, outro é IA. Tente descobrir qual é qual!\n")

        perguntas = [
            "Oi, como você está?",
            "Qual seu nome?",
            "Você é humano ou máquina?",
            "Quanto é 123 x 456?",
            "O que você fez hoje?",
            "Você sonha à noite?",
            "Qual sua comida favorita?",
        ]

        for rodada in range(min(num_rodadas, len(perguntas))):
            pergunta = perguntas[rodada]

            print(f"\n--- RODADA {rodada + 1} ---")
            print(f"Avaliador: {pergunta}\n")

            # Ambos respondem
            for i, agente in enumerate(self.agentes):
                letra = "A" if i == 0 else "B"
                resposta = agente.responder(pergunta, adicionar_delay=False)
                print(f"{letra}: {resposta}")

                self.historico.append({
                    "rodada": rodada + 1,
                    "pergunta": pergunta,
                    "agente": letra,
                    "tipo": "IA" if isinstance(agente, ChatbotSimples) else "Humano",
                    "resposta": resposta
                })

        print(f"\n{'='*70}")

    def revelar_resultado(self) -> Tuple[str, str]:
        """Revelar qual agente é qual"""
        agente_a_tipo = "IA" if isinstance(self.agentes[0], ChatbotSimples) else "Humano"
        agente_b_tipo = "Humano" if agente_a_tipo == "IA" else "IA"

        return agente_a_tipo, agente_b_tipo

    def analisar_caracteristicas(self):
        """Analisar características das respostas"""
        print(f"\n{'='*70}")
        print("ANÁLISE DE CARACTERÍSTICAS")
        print(f"{'='*70}")

        print("\n🔍 PISTAS PARA IDENTIFICAR IA vs HUMANO:\n")

        print("🤖 CARACTERÍSTICAS DE IA:")
        print("   ✓ Respostas muito consistentes e rápidas")
        print("   ✓ Gramática perfeita (a menos que simulem erros)")
        print("   ✓ Evitam compartilhar detalhes pessoais específicos")
        print("   ✓ Respostas genéricas e 'seguras'")
        print("   ✓ Podem calcular muito rápido")

        print("\n👤 CARACTERÍSTICAS HUMANAS:")
        print("   ✓ Variação no tempo de resposta")
        print("   ✓ Erros de digitação naturais")
        print("   ✓ Compartilham experiências pessoais")
        print("   ✓ Usam gírias e expressões coloquiais")
        print("   ✓ Às vezes se distraem ou mudam de assunto")

# ═══════════════════════════════════════════════════════════════════
# EXECUÇÃO DO TESTE
# ═══════════════════════════════════════════════════════════════════

teste = TestedeTuring()
teste.executar_conversa(num_rodadas=7)

# Pedir palpite do usuário
print(f"\n{'='*70}")
print("MOMENTO DA VERDADE")
print(f"{'='*70}")

print("\n🤔 Baseado nas conversas acima, qual seu palpite?")
print("\nAgente A é: (1) Humano  (2) IA")
print("Agente B é: (1) Humano  (2) IA")

# Simulação do palpite (em produção, seria input())
palpite_a = random.choice(["Humano", "IA"])
palpite_b = "IA" if palpite_a == "Humano" else "Humano"

print(f"\n👤 Seu palpite:")
print(f"   Agente A: {palpite_a}")
print(f"   Agente B: {palpite_b}")

# Revelar resultado real
real_a, real_b = teste.revelar_resultado()

print(f"\n✅ RESPOSTA CORRETA:")
print(f"   Agente A: {real_a}")
print(f"   Agente B: {real_b}")

# Verificar acerto
acertou_a = (palpite_a == real_a)
acertou_b = (palpite_b == real_b)
acertos = acertou_a + acertou_b

print(f"\n📊 RESULTADO: {acertos}/2 acertos")

# Interpretação do Teste de Turing
print(f"\n{'='*70}")
print("INTERPRETAÇÃO DO TESTE DE TURING")
print(f"{'='*70}")

if acertos == 2:
    print("\n✅ VOCÊ IDENTIFICOU CORRETAMENTE AMBOS!")
    print("   → IA NÃO passou no Teste de Turing")
    print("   → Foi fácil distinguir máquina de humano")
    print("   → A IA ainda tem características artificiais óbvias")
    print("\n💡 No Teste original, a IA só 'passa' se ENGANAR o avaliador!")
elif acertos == 1:
    print("\n😐 VOCÊ ACERTOU APENAS UM")
    print("   → IA teve desempenho PARCIAL no Teste de Turing")
    print("   → Conseguiu enganar em 50% das identificações")
    print("   → Está no caminho, mas ainda não passa completamente")
elif acertos == 0:
    print("\n🎉 VOCÊ ERROU AMBOS!")
    print("   → IA PASSOU no Teste de Turing! 🤖")
    print("   → A IA conseguiu enganar completamente o avaliador")
    print("   → Você não conseguiu distinguir máquina de humano")
    print("\n💡 Este é o objetivo: a IA imita tão bem que não é detectada!")

# Análise
teste.analisar_caracteristicas()

# ═══════════════════════════════════════════════════════════════════
# DISCUSSÃO SOBRE O TESTE DE TURING
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("DISCUSSÃO: O TESTE DE TURING AINDA É RELEVANTE?")
print(f"{'='*70}")

print("\n✅ ARGUMENTOS A FAVOR:")
print("   • Teste prático e objetivo")
print("   • Foca no comportamento, não na implementação")
print("   • ChatGPT e GPT-4 passam em versões do teste")

print("\n❌ CRÍTICAS:")
print("   • Imitar humanos ≠ Ser inteligente")
print("   • IA pode ser inteligente de forma diferente")
print("   • Não testa criatividade, consciência, emoções")
print("   • 'Chinese Room' (Searle): Sintaxe ≠ Semântica")

print("\n🔮 TESTES MODERNOS ALTERNATIVOS:")
print("   • Winograd Schema (raciocínio de senso comum)")
print("   • Benchmarks especializados (SuperGLUE, MMLU)")
print("   • Testes de criatividade e abstração")
print("   • Avaliação ética e viés")

print("\n💡 CONCLUSÃO:")
print("O Teste de Turing foi revolucionário em 1950, mas hoje sabemos")
print("que 'passar no teste' não significa ter consciência ou compreensão.")
print("LLMs modernos passam no teste, mas ainda não são AGI.")

print("\n✅ EXERCÍCIO COMPLETO!")
