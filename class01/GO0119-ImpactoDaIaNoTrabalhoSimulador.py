# GO0119-ImpactoDaIaNoTrabalhoSimulador
import numpy as np
import matplotlib.pyplot as plt
import random

# ═══════════════════════════════════════════════════════════════════
# IMPACTO DA IA NO TRABALHO - SIMULADOR DE AUTOMAÇÃO
# ═══════════════════════════════════════════════════════════════════

print("="*70)
print("FUTURO DO TRABALHO: IMPACTO DA IA")
print("="*70)

# Base de dados: Profissões e risco de automação
# Baseado em estudos Frey & Osborne (2013), McKinsey (2023)
profissoes = [
    {
        "nome": "Motorista de Caminhão",
        "risco_automacao": 95,  # %
        "tarefas_automatizaveis": ["Dirigir", "Navegação", "Frenagem"],
        "tarefas_humanas": ["Emergências", "Relações com clientes"],
        "salario_medio": 3000,
        "trabalhadores_br": 2000000,
        "tecnologia": "Veículos autônomos (Tesla, Waymo)"
    },
    {
        "nome": "Caixa de Supermercado",
        "risco_automacao": 90,
        "tarefas_automatizaveis": ["Escanear produtos", "Processar pagamento"],
        "tarefas_humanas": ["Atendimento especial", "Resolução de problemas"],
        "salario_medio": 1500,
        "trabalhadores_br": 1500000,
        "tecnologia": "Self-checkout, Amazon Go"
    },
    {
        "nome": "Operador de Telemarketing",
        "risco_automacao": 99,
        "tarefas_automatizaveis": ["Atendimento padrão", "Vendas script", "Agendamento"],
        "tarefas_humanas": ["Casos complexos", "Empatia em reclamações"],
        "salario_medio": 1800,
        "trabalhadores_br": 1200000,
        "tecnologia": "Chatbots (GPT-4, Dialogflow)"
    },
    {
        "nome": "Contador",
        "risco_automacao": 60,
        "tarefas_automatizaveis": ["Lançamentos contábeis", "Conciliação bancária", "Folha de pagamento"],
        "tarefas_humanas": ["Planejamento tributário", "Consultoria estratégica"],
        "salario_medio": 5000,
        "trabalhadores_br": 800000,
        "tecnologia": "Software de contabilidade (QuickBooks AI)"
    },
    {
        "nome": "Desenvolvedor de Software",
        "risco_automacao": 40,
        "tarefas_automatizaveis": ["Código boilerplate", "Testes", "Debug simples"],
        "tarefas_humanas": ["Arquitetura", "Requisitos", "Criatividade"],
        "salario_medio": 8000,
        "trabalhadores_br": 500000,
        "tecnologia": "GitHub Copilot, GPT-4"
    },
    {
        "nome": "Médico Radiologista",
        "risco_automacao": 70,
        "tarefas_automatizaveis": ["Análise de imagens", "Detecção de anomalias"],
        "tarefas_humanas": ["Diagnóstico complexo", "Interação com paciente"],
        "salario_medio": 15000,
        "trabalhadores_br": 30000,
        "tecnologia": "Deep learning (ResNet, U-Net)"
    },
    {
        "nome": "Professor",
        "risco_automacao": 25,
        "tarefas_automatizaveis": ["Correção de provas", "Conteúdo padronizado"],
        "tarefas_humanas": ["Mentoria", "Adaptação pedagógica", "Empatia"],
        "salario_medio": 4000,
        "trabalhadores_br": 2500000,
        "tecnologia": "Plataformas adaptativas (Coursera, Khan Academy)"
    },
    {
        "nome": "Artista/Designer",
        "risco_automacao": 30,
        "tarefas_automatizaveis": ["Design simples", "Variações", "Otimização"],
        "tarefas_humanas": ["Criatividade original", "Conceito", "Emoção"],
        "salario_medio": 4500,
        "trabalhadores_br": 400000,
        "tecnologia": "DALL-E, Midjourney, Stable Diffusion"
    },
    {
        "nome": "Psicólogo/Terapeuta",
        "risco_automacao": 15,
        "tarefas_automatizaveis": ["Triagem inicial", "Questionários"],
        "tarefas_humanas": ["Empatia profunda", "Terapia complexa", "Confiança"],
        "salario_medio": 6000,
        "trabalhadores_br": 300000,
        "tecnologia": "Chatbots de saúde mental (Woebot)"
    },
    {
        "nome": "Operador de Linha de Produção",
        "risco_automacao": 85,
        "tarefas_automatizaveis": ["Montagem repetitiva", "Controle qualidade visual"],
        "tarefas_humanas": ["Ajustes finos", "Manutenção"],
        "salario_medio": 2500,
        "trabalhadores_br": 3000000,
        "tecnologia": "Robôs industriais (ABB, FANUC)"
    }
]

print("\n📊 ANÁLISE DE PROFISSÕES:")
print("="*70)

# Ordenar por risco
profissoes_ordenadas = sorted(profissoes, key=lambda p: p['risco_automacao'], reverse=True)

for i, prof in enumerate(profissoes_ordenadas, 1):
    risco = prof['risco_automacao']
    if risco >= 80:
        emoji = "🔴"
        nivel = "ALTO"
    elif risco >= 50:
        emoji = "🟡"
        nivel = "MÉDIO"
    else:
        emoji = "🟢"
        nivel = "BAIXO"

    print(f"\n{i}. {emoji} {prof['nome']}")
    print(f"   Risco de Automação: {risco}% ({nivel})")
    print(f"   Trabalhadores no BR: {prof['trabalhadores_br']:,}")
    print(f"   Tecnologia: {prof['tecnologia']}")
    print(f"   Tarefas automatizáveis: {', '.join(prof['tarefas_automatizaveis'][:2])}")
    print(f"   Tarefas humanas: {', '.join(prof['tarefas_humanas'][:2])}")

# Calcular impacto total
print("\n" + "="*70)
print("IMPACTO TOTAL NO BRASIL")
print("="*70)

total_trabalhadores = sum(p['trabalhadores_br'] for p in profissoes)
trabalhadores_alto_risco = sum(p['trabalhadores_br'] for p in profissoes 
                                if p['risco_automacao'] >= 70)

percentual_risco = (trabalhadores_alto_risco / total_trabalhadores) * 100

print(f"\n📈 Total analisado: {total_trabalhadores:,} trabalhadores")
print(f"🔴 Alto risco (≥70%): {trabalhadores_alto_risco:,} trabalhadores")
print(f"⚠️ Percentual em risco: {percentual_risco:.1f}%")

# Simulação: Cenários de automação
print("\n" + "="*70)
print("SIMULAÇÃO: 3 CENÁRIOS (2026-2036)")
print("="*70)

cenarios = {
    "Otimista": {
        "descricao": "Automação lenta, requalificação eficaz",
        "taxa_automacao": 0.3,  # 30% das tarefas automatizadas em 10 anos
        "taxa_requalificacao": 0.7,  # 70% conseguem novo emprego
        "novos_empregos": 1.2  # 20% a mais de empregos criados que perdidos
    },
    "Realista": {
        "descricao": "Automação moderada, requalificação parcial",
        "taxa_automacao": 0.5,
        "taxa_requalificacao": 0.4,
        "novos_empregos": 0.8  # 20% a menos
    },
    "Pessimista": {
        "descricao": "Automação rápida, pouca requalificação",
        "taxa_automacao": 0.7,
        "taxa_requalificacao": 0.2,
        "novos_empregos": 0.5  # 50% a menos
    }
}

resultados_cenarios = {}

for nome_cenario, params in cenarios.items():
    print(f"\n🔮 CENÁRIO {nome_cenario.upper()}:")
    print(f"   {params['descricao']}")

    empregos_perdidos = 0
    empregos_criados = 0
    desempregados_finais = 0

    for prof in profissoes:
        # Calcular perda
        risco = prof['risco_automacao'] / 100
        perda = prof['trabalhadores_br'] * risco * params['taxa_automacao']
        empregos_perdidos += perda

        # Calcular requalificação
        requalificados = perda * params['taxa_requalificacao']
        desempregados = perda - requalificados
        desempregados_finais += desempregados

    # Novos empregos criados
    empregos_criados = empregos_perdidos * params['novos_empregos']

    # Desemprego líquido
    desemprego_liquido = desempregados_finais - empregos_criados

    print(f"\n   📉 Empregos perdidos: {empregos_perdidos:,.0f}")
    print(f"   📈 Novos empregos (IA): {empregos_criados:,.0f}")
    print(f"   ✅ Requalificados: {empregos_perdidos * params['taxa_requalificacao']:,.0f}")
    print(f"   ❌ Desempregados finais: {desemprego_liquido:,.0f}")

    taxa_desemprego = (desemprego_liquido / total_trabalhadores) * 100
    print(f"   📊 Taxa desemprego adicional: {taxa_desemprego:+.1f}%")

    resultados_cenarios[nome_cenario] = {
        "perdidos": empregos_perdidos,
        "criados": empregos_criados,
        "desemprego": desemprego_liquido,
        "taxa": taxa_desemprego
    }

# Recomendações
print("\n" + "="*70)
print("RECOMENDAÇÕES PARA TRABALHADORES")
print("="*70)

recomendacoes = {
    "Alto Risco (70-100%)": [
        "🎓 Requalificação URGENTE",
        "🔄 Buscar tarefas não-automatizáveis",
        "💼 Considerar transição de carreira",
        "📚 Aprender habilidades de IA (como usá-la, não competir)"
    ],
    "Médio Risco (40-70%)": [
        "📖 Educação continuada",
        "🤝 Focar em tarefas humanas (relacional, criativo)",
        "🛠️ Aprender a usar ferramentas de IA (aumentar produtividade)",
        "🌐 Networking e soft skills"
    ],
    "Baixo Risco (<40%)": [
        "✨ Continuar desenvolvendo criatividade e empatia",
        "🤖 Usar IA como assistente (não ameaça)",
        "🎯 Especialização em nichos",
        "👥 Fortalecer habilidades interpessoais"
    ]
}

for categoria, dicas in recomendacoes.items():
    print(f"\n{categoria}:")
    for dica in dicas:
        print(f"   • {dica}")

# Visualização
print("\n📊 GERANDO VISUALIZAÇÕES...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Risco de automação por profissão
ax1 = axes[0, 0]
nomes = [p['nome'][:20] for p in profissoes_ordenadas]
riscos = [p['risco_automacao'] for p in profissoes_ordenadas]
cores = ['red' if r >= 70 else 'orange' if r >= 40 else 'green' for r in riscos]

bars = ax1.barh(nomes, riscos, color=cores, alpha=0.7)
ax1.set_xlabel("Risco de Automação (%)", fontsize=12)
ax1.set_title("Profissões em Risco", fontsize=13, fontweight='bold')
ax1.axvline(x=70, color='red', linestyle='--', label='Alto Risco (70%)')
ax1.axvline(x=40, color='orange', linestyle='--', label='Médio Risco (40%)')
ax1.legend()
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3, axis='x')

# 2. Impacto por número de trabalhadores
ax2 = axes[0, 1]
trabalhadores = [p['trabalhadores_br'] / 1000 for p in profissoes_ordenadas]  # Em milhares
ax2.scatter(riscos, trabalhadores, s=[r*5 for r in riscos], c=cores, alpha=0.6)

for i, prof in enumerate(profissoes_ordenadas):
    ax2.annotate(prof['nome'][:15], 
                (riscos[i], trabalhadores[i]),
                fontsize=7, alpha=0.7)

ax2.set_xlabel("Risco de Automação (%)", fontsize=12)
ax2.set_ylabel("Trabalhadores (milhares)", fontsize=12)
ax2.set_title("Risco vs Impacto Social", fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Cenários de futuro
ax3 = axes[1, 0]
cenarios_nomes = list(resultados_cenarios.keys())
desempregos = [resultados_cenarios[c]['desemprego']/1000 for c in cenarios_nomes]
cores_cenarios = ['green', 'orange', 'red']

bars = ax3.bar(cenarios_nomes, desempregos, color=cores_cenarios, alpha=0.7)
ax3.set_ylabel("Desemprego Líquido (milhares)", fontsize=12)
ax3.set_title("Impacto em 3 Cenários (2026-2036)", fontsize=13, fontweight='bold')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax3.grid(True, alpha=0.3, axis='y')

# Adicionar valores nas barras
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{desempregos[i]:,.0f}k',
            ha='center', va='bottom' if height > 0 else 'top', fontsize=10)

# 4. Habilidades do futuro
ax4 = axes[1, 1]
ax4.axis('off')

texto_futuro = """
🚀 HABILIDADES DO FUTURO

ALTO VALOR (Difícil de Automatizar):
✅ Criatividade e inovação
✅ Inteligência emocional e empatia
✅ Pensamento crítico complexo
✅ Resolução de problemas não-estruturados
✅ Liderança e influência
✅ Negociação e persuasão

MÉDIO VALOR (IA como Assistente):
🤝 Uso de ferramentas de IA
🤝 Análise de dados
🤝 Gerenciamento de projetos
🤝 Comunicação técnica

BAIXO VALOR (Automação Iminente):
❌ Tarefas repetitivas
❌ Seguir scripts
❌ Processamento de dados simples
❌ Atendimento padrão

PROFISSÕES EMERGENTES (IA):
🆕 Prompt Engineer
🆕 AI Ethics Specialist
🆕 AI Trainer/Curator
🆕 Human-AI Interaction Designer
🆕 AI Auditor

ESTRATÉGIA: T-shaped Skills
   Profundidade (expertise) + Amplitude (cross-functional)
"""

ax4.text(0.1, 0.5, texto_futuro, fontsize=9, family='monospace',
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax4.set_title("Adaptação ao Futuro", fontsize=13, fontweight='bold')

plt.suptitle("Impacto da IA no Futuro do Trabalho", 
            fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()

# Conclusão
print("\n💡 CONCLUSÕES:")
print("="*70)
print("1️⃣ 47% dos empregos em risco nas próximas décadas (Frey & Osborne)")
print("2️⃣ Tarefas repetitivas e regradas = Alto risco")
print("3️⃣ Criatividade, empatia, pensamento crítico = Baixo risco")
print("4️⃣ Requalificação é essencial (aprendizado contínuo)")
print("5️⃣ IA cria novos empregos MAS requer novas habilidades")

print("\n🎯 MENSAGEM PARA ESTUDANTES:")
print("   → Não competir COM a IA, mas aprender a trabalhar com ela")
print("   → Focar em habilidades humanas únicas")
print("   → Educação continuada para toda vida (lifelong learning)")
print("   → IA é ferramenta, não substituto completo")

print("\n✅ SIMULAÇÃO COMPLETA!")
