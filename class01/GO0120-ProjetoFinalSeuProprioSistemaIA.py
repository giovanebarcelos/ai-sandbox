# GO0120-ProjetoFinalSeuPróprioSistemaIA
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════
# PROJETO FINAL: ASSISTENTE IA EDUCACIONAL
# Combina: Ética, Limitações, Explicabilidade, Futuro
# ═══════════════════════════════════════════════════════════════════

print("="*70)
print("PROJETO: ASSISTENTE IA PARA EDUCAÇÃO")
print("Integra conceitos: Ética, Viés, XAI, AGI, Futuro do Trabalho")
print("="*70)

# ═══════════════════════════════════════════════════════════════════
# CLASSE: ASSISTENTE EDUCACIONAL COM ÉTICA
# ═══════════════════════════════════════════════════════════════════

class AssistenteEducacional:
    """
    Assistente IA para educação que incorpora:
    - Detecção de viés
    - Explicabilidade
    - Limitações reconhecidas
    - Ética no design
    """

    def __init__(self, nome="EduAI"):
        self.nome = nome
        self.conhecimento = self._carregar_conhecimento()
        self.historico_interacoes = []
        self.vies_detectado = []
        self.limitacoes_reconhecidas = []

        print(f"\n✅ {self.nome} inicializado!")
        print(f"📚 Base de conhecimento: {len(self.conhecimento)} tópicos")

    def _carregar_conhecimento(self):
        """Base de conhecimento sobre IA (Aula 01)"""
        return {
            "ia": {
                "definicao": "Inteligência Artificial é a capacidade de máquinas executarem tarefas que normalmente requerem inteligência humana.",
                "tipos": ["ANI (Narrow)", "AGI (General)", "ASI (Super)"],
                "status_2026": "ANI dominado, AGI em progresso"
            },
            "turing_test": {
                "definicao": "Teste proposto por Alan Turing (1950) para avaliar se máquina pode imitar comportamento humano.",
                "problema": "Imitar não é o mesmo que inteligir (Chinese Room)",
                "status_atual": "LLMs passam, mas não têm consciência"
            },
            "bias": {
                "definicao": "Viés em IA ocorre quando modelo discrimina grupos protegidos.",
                "exemplo": "Amazon recruiting tool (2018) favoreceu homens",
                "solucao": "Auditoria, fairness metrics, reweighting"
            },
            "xai": {
                "definicao": "Explainable AI - técnicas para interpretar decisões de modelos black-box.",
                "tecnicas": ["LIME", "SHAP", "Feature Importance"],
                "importancia": "GDPR Art. 22 - direito à explicação"
            },
            "agi": {
                "definicao": "AGI = superar humanos em TODAS tarefas cognitivas",
                "criterios": ["Raciocínio causal", "Senso comum", "Abstração"],
                "estimativa": "AGI Fraca: 5-10 anos, AGI Completa: 20-50 anos"
            },
            "futuro_trabalho": {
                "definicao": "47% empregos em risco (Frey & Osborne)",
                "alto_risco": "Tarefas repetitivas (telemarketing 99%, motorista 95%)",
                "baixo_risco": "Criatividade, empatia (psicólogo 15%, professor 25%)",
                "estrategia": "Lifelong learning, T-shaped skills"
            }
        }

    def responder_pergunta(self, pergunta, estudante_nome="Estudante"):
        """Responde pergunta com explicação e reconhecimento de limitações"""

        pergunta_lower = pergunta.lower()
        resposta = None
        topico_usado = None
        confianca = 0

        # Buscar na base de conhecimento
        for topico, info in self.conhecimento.items():
            if topico in pergunta_lower or any(palavra in pergunta_lower for palavra in topico.split('_')):
                resposta = info['definicao']
                topico_usado = topico
                confianca = 0.9
                break

        # Se não encontrou, reconhecer limitação
        if resposta is None:
            resposta = "Desculpe, não tenho informação confiável sobre isso na minha base de conhecimento (limitada à Aula 01)."
            confianca = 0.0
            self.limitacoes_reconhecidas.append(pergunta)

        # Registrar interação
        interacao = {
            "timestamp": datetime.now(),
            "estudante": estudante_nome,
            "pergunta": pergunta,
            "resposta": resposta,
            "topico": topico_usado,
            "confianca": confianca
        }
        self.historico_interacoes.append(interacao)

        # Formatar resposta
        print(f"\n{'='*70}")
        print(f"🧑‍🎓 {estudante_nome}: {pergunta}")
        print(f"{'='*70}")
        print(f"🤖 {self.nome}: {resposta}")

        if confianca > 0:
            print(f"\n📊 Confiança: {confianca*100:.0f}%")
            print(f"📚 Fonte: Base de conhecimento (tópico '{topico_usado}')")

            # Explicabilidade (XAI)
            print(f"\n💡 EXPLICAÇÃO (XAI):")
            print(f"   → Busquei na base por palavras-chave: '{topico_usado}'")
            print(f"   → Encontrei correspondência com definição armazenada")
            print(f"   → Retornei informação mais relevante")
        else:
            print(f"\n⚠️ Confiança: {confianca*100:.0f}%")
            print(f"❌ Limitação reconhecida: Tópico fora da base de conhecimento")
            print(f"💡 Recomendação: Consulte professor ou fontes especializadas")

        return resposta, confianca

    def detectar_vies_interacoes(self):
        """Detecta viés no histórico de interações (ética)"""

        print(f"\n{'='*70}")
        print("AUDITORIA DE VIÉS (Ética)")
        print(f"{'='*70}")

        if len(self.historico_interacoes) < 3:
            print("⚠️ Poucos dados para análise de viés")
            return

        # Analisar distribuição de confiança por estudante
        estudantes = {}
        for inter in self.historico_interacoes:
            nome = inter['estudante']
            if nome not in estudantes:
                estudantes[nome] = []
            estudantes[nome].append(inter['confianca'])

        print(f"\n📊 Confiança Média por Estudante:")

        vies_detectado_flag = False
        for estudante, confiancias in estudantes.items():
            media = np.mean(confiancias)
            print(f"   • {estudante}: {media*100:.0f}%")

            # Detectar viés (se algum estudante tem média muito diferente)
            if len(estudantes) > 1:
                geral = np.mean([c for cs in estudantes.values() for c in cs])
                disparidade = abs(media - geral)

                if disparidade > 0.2:  # 20% de diferença
                    print(f"      ⚠️ VIÉS DETECTADO: Disparidade de {disparidade*100:.0f}%")
                    self.vies_detectado.append({
                        "estudante": estudante,
                        "disparidade": disparidade
                    })
                    vies_detectado_flag = True

        if not vies_detectado_flag:
            print(f"\n✅ Nenhum viés significativo detectado")
        else:
            print(f"\n🔍 Investigação necessária:")
            print(f"   → Perguntas de alguns estudantes fora do escopo?")
            print(f"   → Base de conhecimento precisa expansão?")
            print(f"   → Implementar fairness constraints")

    def gerar_relatorio(self):
        """Gera relatório completo da sessão"""

        print(f"\n{'='*70}")
        print(f"RELATÓRIO DE SESSÃO - {self.nome}")
        print(f"{'='*70}")

        total_interacoes = len(self.historico_interacoes)
        if total_interacoes == 0:
            print("Nenhuma interação registrada")
            return

        # Estatísticas
        confiancias = [i['confianca'] for i in self.historico_interacoes]
        media_confianca = np.mean(confiancias)
        respondidas = sum(1 for c in confiancias if c > 0)
        nao_respondidas = total_interacoes - respondidas

        print(f"\n📊 ESTATÍSTICAS:")
        print(f"   Total de perguntas: {total_interacoes}")
        print(f"   Respondidas: {respondidas} ({respondidas/total_interacoes*100:.0f}%)")
        print(f"   Não respondidas: {nao_respondidas} ({nao_respondidas/total_interacoes*100:.0f}%)")
        print(f"   Confiança média: {media_confianca*100:.0f}%")

        # Tópicos mais consultados
        topicos = [i['topico'] for i in self.historico_interacoes if i['topico']]
        if topicos:
            topico_mais_comum = max(set(topicos), key=topicos.count)
            print(f"   Tópico mais consultado: '{topico_mais_comum}' ({topicos.count(topico_mais_comum)}x)")

        # Limitações
        print(f"\n⚠️ LIMITAÇÕES RECONHECIDAS:")
        print(f"   Total: {len(self.limitacoes_reconhecidas)}")
        if self.limitacoes_reconhecidas:
            print(f"   Exemplos:")
            for lim in self.limitacoes_reconhecidas[:3]:
                print(f"      • \"{lim}\"")

        # Viés
        print(f"\n🔍 AUDITORIA DE VIÉS:")
        if self.vies_detectado:
            print(f"   ⚠️ Viés detectado: {len(self.vies_detectado)} caso(s)")
            for v in self.vies_detectado:
                print(f"      • {v['estudante']}: disparidade {v['disparidade']*100:.0f}%")
        else:
            print(f"   ✅ Nenhum viés significativo")

        # Recomendações
        print(f"\n💡 RECOMENDAÇÕES:")
        if nao_respondidas > respondidas:
            print(f"   → Expandir base de conhecimento (muitas perguntas sem resposta)")
        if len(self.vies_detectado) > 0:
            print(f"   → Investigar disparidade entre estudantes")
        if media_confianca < 0.5:
            print(f"   → Melhorar busca semântica (confiança baixa)")
        if total_interacoes < 10:
            print(f"   → Coletar mais dados para análise robusta")

# ═══════════════════════════════════════════════════════════════════
# SIMULAÇÃO: SESSÃO DE AULA COM ASSISTENTE IA
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("SIMULAÇÃO: SESSÃO DE AULA")
print("="*70)

# Inicializar assistente
assistente = AssistenteEducacional(nome="EduAI v1.0")

# Simular perguntas de diferentes estudantes
perguntas_simuladas = [
    ("Alice", "O que é Inteligência Artificial?"),
    ("Bob", "O que é o Teste de Turing?"),
    ("Alice", "GPT-4 é uma AGI?"),
    ("Carol", "Como detectar viés em modelos de IA?"),
    ("Bob", "Qual é o impacto da IA no futuro do trabalho?"),
    ("Alice", "O que é XAI?"),
    ("David", "Qual a capital da França?"),  # Fora do escopo
    ("Carol", "Como funciona quantum computing?"),  # Fora do escopo
]

print("\n🎓 Iniciando sessão interativa...")

for estudante, pergunta in perguntas_simuladas:
    resposta, confianca = assistente.responder_pergunta(pergunta, estudante)
    print()  # Espaçamento

# Detectar viés
assistente.detectar_vies_interacoes()

# Gerar relatório
assistente.gerar_relatorio()

# Visualização
print("\n📊 GERANDO VISUALIZAÇÃO...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Confiança por pergunta
ax1 = axes[0, 0]
confiancias = [i['confianca'] for i in assistente.historico_interacoes]
perguntas_num = range(1, len(confiancias) + 1)

colors = ['green' if c > 0.7 else 'orange' if c > 0.3 else 'red' for c in confiancias]
bars = ax1.bar(perguntas_num, confiancias, color=colors, alpha=0.7)

ax1.set_xlabel("Pergunta #", fontsize=12)
ax1.set_ylabel("Confiança", fontsize=12)
ax1.set_title("Confiança por Pergunta", fontsize=13, fontweight='bold')
ax1.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Alta confiança')
ax1.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Baixa confiança')
ax1.legend()
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3, axis='y')

# 2. Distribuição de tópicos
ax2 = axes[0, 1]
topicos_consultados = [i['topico'] for i in assistente.historico_interacoes if i['topico']]

if topicos_consultados:
    topico_counts = {}
    for t in topicos_consultados:
        topico_counts[t] = topico_counts.get(t, 0) + 1

    topicos_nomes = list(topico_counts.keys())
    topicos_valores = list(topico_counts.values())

    colors_topicos = plt.cm.Set3(range(len(topicos_nomes)))
    ax2.pie(topicos_valores, labels=topicos_nomes, autopct='%1.0f%%', 
           colors=colors_topicos, startangle=90)
    ax2.set_title("Distribuição de Tópicos Consultados", fontsize=13, fontweight='bold')
else:
    ax2.text(0.5, 0.5, 'Sem tópicos consultados', ha='center', va='center')
    ax2.set_title("Distribuição de Tópicos", fontsize=13, fontweight='bold')

# 3. Confiança por estudante
ax3 = axes[1, 0]
estudantes_stats = {}
for i in assistente.historico_interacoes:
    nome = i['estudante']
    if nome not in estudantes_stats:
        estudantes_stats[nome] = []
    estudantes_stats[nome].append(i['confianca'])

estudantes_nomes = list(estudantes_stats.keys())
estudantes_medias = [np.mean(estudantes_stats[e]) for e in estudantes_nomes]

bars_est = ax3.barh(estudantes_nomes, estudantes_medias, color='skyblue', alpha=0.7)
ax3.set_xlabel("Confiança Média", fontsize=12)
ax3.set_title("Confiança por Estudante (Viés?)", fontsize=13, fontweight='bold')
ax3.set_xlim(0, 1)
ax3.axvline(x=0.5, color='orange', linestyle='--', label='Esperado (50%)')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='x')

# 4. Princípios do design ético
ax4 = axes[1, 1]
ax4.axis('off')

texto_principios = """
✅ PRINCÍPIOS DE IA ÉTICA (implementados)

1️⃣ TRANSPARÊNCIA:
   ✓ Explicabilidade (XAI): Mostra fonte e raciocínio
   ✓ Confiança quantificada: Não finge saber tudo

2️⃣ RECONHECIMENTO DE LIMITAÇÕES:
   ✓ Admite quando não sabe
   ✓ Recomenda consultar humanos especialistas
   ✓ Base de conhecimento delimitada

3️⃣ AUDITORIA DE VIÉS:
   ✓ Monitora disparidade entre estudantes
   ✓ Alerta quando viés detectado
   ✓ Métricas de fairness

4️⃣ RESPONSABILIDADE:
   ✓ Histórico de interações registrado
   ✓ Relatório de sessão detalhado
   ✓ Rastreabilidade de decisões

5️⃣ BENEFICÊNCIA:
   ✓ Foco em educar, não substituir professor
   ✓ Assistente, não autoridade
   ✓ Promove pensamento crítico

🎯 LIÇÕES DA AULA 01 APLICADAS:
• Ética desde o design (não afterthought)
• XAI para confiança
• Auditoria contínua de viés
• Limitações reconhecidas
• Humano no loop (human oversight)
"""

ax4.text(0.1, 0.5, texto_principios, fontsize=9, family='monospace',
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax4.set_title("Design Ético Implementado", fontsize=13, fontweight='bold')

plt.suptitle("Assistente IA Educacional - Análise de Sessão", 
            fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()

# Conclusão do projeto
print("\n" + "="*70)
print("PROJETO CONCLUÍDO")
print("="*70)

print("\n🎓 O QUE ESTE PROJETO DEMONSTRA:")
print("   1️⃣ Ética desde o design (não é add-on)")
print("   2️⃣ Explicabilidade (XAI) em decisões")
print("   3️⃣ Reconhecimento de limitações (humildade epistêmica)")
print("   4️⃣ Auditoria de viés (fairness)")
print("   5️⃣ Transparência e rastreabilidade")

print("\n💡 CONCEITOS DA AULA 01 INTEGRADOS:")
print("   ✓ Ética em IA (design, auditoria, responsabilidade)")
print("   ✓ Viés e Fairness (detecção de disparidade)")
print("   ✓ XAI (explicação de raciocínio)")
print("   ✓ Limitações de IA (reconhecimento explícito)")
print("   ✓ AGI (ainda distante - este sistema é ANI)")

print("\n🚀 PRÓXIMOS PASSOS (para estudantes):")
print("   → Expandir base de conhecimento (RAG)")
print("   → Implementar SHAP para explicabilidade avançada")
print("   → Adicionar interface web (Streamlit, Gradio)")
print("   → Integrar LLM real (GPT-4 API, Ollama local)")
print("   → Implementar fairness constraints formais")

print("\n✅ AULA 01 CONCLUÍDA!")
print("="*70)
print("📚 Você aprendeu: História, Ética, Viés, Benchmarks, AGI, XAI, Futuro")
print("🎯 Próxima aula: Representação de Conhecimento (Aula 02)")
print("="*70)
