# GO0116-ComparaçãoIaSimbólicaVsConexionista
import numpy as np
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════
# COMPARAÇÃO: IA SIMBÓLICA vs CONEXIONISTA
# ═══════════════════════════════════════════════════════════════════

print("="*70)
print("PARADIGMAS DE IA: SIMBÓLICA vs CONEXIONISTA")
print("="*70)

# Comparação lado a lado
comparacao = {
    "Representação": {
        "Simbólica": "Símbolos e regras explícitas\n(IF-THEN, lógica, ontologias)",
        "Conexionista": "Pesos em redes neurais\n(números, vetores, matrizes)"
    },
    "Aprendizado": {
        "Simbólica": "Programação manual\n(especialistas codificam conhecimento)",
        "Conexionista": "Automático a partir de dados\n(backpropagation, gradient descent)"
    },
    "Interpretabilidade": {
        "Simbólica": "✅ Alta (regras legíveis)\n'SE febre E tosse ENTÃO gripe'",
        "Conexionista": "❌ Baixa (black-box)\n'Neurônio 347 ativou 0.82'"
    },
    "Generalização": {
        "Simbólica": "❌ Frágil (não lida com novidades)\nSó funciona em domínio estreito",
        "Conexionista": "✅ Robusta (padrões estatísticos)\nLida bem com ruído e variações"
    },
    "Dados": {
        "Simbólica": "Pouco dado (conhecimento explícito)\n10-100 regras",
        "Conexionista": "Muito dado necessário\n1M-1B exemplos"
    },
    "Exemplos": {
        "Simbólica": "Sistemas especialistas (MYCIN)\nChess engines (Stockfish)\nProlog, Answer Set Programming",
        "Conexionista": "Deep Learning (GPT, BERT)\nVision (ResNet, YOLO)\nAlphaGo"
    },
    "Limitações": {
        "Simbólica": "• Engenharia de conhecimento cara\n• Não escala bem\n• Não lida com ambiguidade",
        "Conexionista": "• Black-box (não explica)\n• Requer muitos dados\n• Pode alucinar"
    }
}

print("\n📊 COMPARAÇÃO DETALHADA:\n")

for aspecto, valores in comparacao.items():
    print(f"{'='*70}")
    print(f"🔹 {aspecto.upper()}")
    print(f"{'='*70}")
    print(f"\n🔵 IA SIMBÓLICA (GOFAI - Good Old-Fashioned AI):")
    print(f"   {valores['Simbólica']}")
    print(f"\n🟢 IA CONEXIONISTA (Subsimbólica, Neural):")
    print(f"   {valores['Conexionista']}")
    print()

# Exemplo prático: Classificar animal
print("\n" + "="*70)
print("EXEMPLO PRÁTICO: CLASSIFICAR ANIMAL")
print("="*70)

print("\n🔵 ABORDAGEM SIMBÓLICA (Regras):")
print("""
def classificar_simbolica(tem_penas, voa, nada):
    if tem_penas and voa:
        return "Pássaro"
    elif tem_penas and nada:
        return "Pinguim"
    elif not tem_penas and nada:
        return "Peixe"
    else:
        return "Desconhecido"

# Exemplo
classificar_simbolica(tem_penas=True, voa=True, nada=False)
# → "Pássaro"

✅ Vantagem: Transparente, explica decisão
❌ Problema: E se animal tem características incomuns?
""")

print("\n🟢 ABORDAGEM CONEXIONISTA (Rede Neural):")
print("""
# Treinar rede neural com 1000 exemplos
modelo = RedeNeural(input=3, hidden=10, output=3)
modelo.fit(X_treino, y_treino)

# Classificar
resultado = modelo.predict([1, 1, 0])  # penas, voa, nada
# → [0.9, 0.05, 0.05] (90% Pássaro)

✅ Vantagem: Aprende automaticamente, lida com variações
❌ Problema: Não explica por quê
""")

# Visualização
print("\n📊 GERANDO VISUALIZAÇÃO...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Linha do tempo
ax1 = axes[0, 0]
anos = [1960, 1980, 2000, 2010, 2020]
popularidade_simbolica = [90, 70, 30, 20, 25]  # Renascimento com XAI
popularidade_conexionista = [10, 30, 70, 80, 75]

ax1.plot(anos, popularidade_simbolica, 'o-', linewidth=3, markersize=10, 
        color='blue', label='Simbólica')
ax1.plot(anos, popularidade_conexionista, 's-', linewidth=3, markersize=10, 
        color='green', label='Conexionista')

ax1.set_xlabel("Ano", fontsize=12)
ax1.set_ylabel("Popularidade Relativa (%)", fontsize=12)
ax1.set_title("Evolução dos Paradigmas de IA", fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 100)

# Anotações
ax1.annotate('Sistemas\nEspecialistas', xy=(1980, 70), xytext=(1985, 85),
            arrowprops=dict(arrowstyle='->', color='blue'),
            fontsize=9, color='blue')
ax1.annotate('Deep Learning\nRevolução', xy=(2010, 80), xytext=(2012, 95),
            arrowprops=dict(arrowstyle='->', color='green'),
            fontsize=9, color='green')

# 2. Comparação de características
ax2 = axes[0, 1]
categorias = ['Interpretabilidade', 'Generalização', 'Escalabilidade', 'Poucos Dados']
simbolica_scores = [9, 3, 4, 8]
conexionista_scores = [3, 9, 9, 2]

x = np.arange(len(categorias))
width = 0.35

bars1 = ax2.barh(x - width/2, simbolica_scores, width, label='Simbólica', 
                color='blue', alpha=0.7)
bars2 = ax2.barh(x + width/2, conexionista_scores, width, label='Conexionista', 
                color='green', alpha=0.7)

ax2.set_yticks(x)
ax2.set_yticklabels(categorias)
ax2.set_xlabel("Score (0-10)", fontsize=12)
ax2.set_title("Comparação de Características", fontsize=13, fontweight='bold')
ax2.legend()
ax2.set_xlim(0, 10)
ax2.grid(True, alpha=0.3, axis='x')

# 3. Aplicações por paradigma
ax3 = axes[1, 0]
aplicacoes_simbolica = ['Sistemas\nEspecialistas', 'Prova de\nTeoremas', 'Planning\n(Robótica)', 'Diagnóstico\nMédico']
aplicacoes_conexionista = ['Visão\nComputacional', 'NLP\n(LLMs)', 'Reconh.\nVoz', 'Jogos\n(AlphaGo)']

y_sim = np.arange(len(aplicacoes_simbolica))
y_con = np.arange(len(aplicacoes_conexionista))

ax3.barh(y_sim + 0.5, [1]*len(aplicacoes_simbolica), height=0.4, 
        color='blue', alpha=0.7, label='Simbólica')
ax3.barh(y_con, [1]*len(aplicacoes_conexionista), height=0.4, 
        color='green', alpha=0.7, label='Conexionista')

all_apps = aplicacoes_conexionista + aplicacoes_simbolica
ax3.set_yticks(range(len(all_apps)))
ax3.set_yticklabels(all_apps, fontsize=9)
ax3.set_xlim(0, 1.2)
ax3.set_title("Principais Aplicações", fontsize=13, fontweight='bold')
ax3.legend()
ax3.set_xticks([])

# 4. Futuro: Neuro-Simbólico
ax4 = axes[1, 1]
ax4.axis('off')

texto_futuro = """
🔮 FUTURO: IA NEURO-SIMBÓLICA

Combina o melhor dos dois mundos:

🧠 NEURAL (Subsimbólico):
   ✓ Percepção (visão, linguagem)
   ✓ Aprendizado de padrões
   ✓ Lida com ruído e ambiguidade

🔗 +

💭 SIMBÓLICO (Raciocínio):
   ✓ Lógica e regras
   ✓ Explicabilidade
   ✓ Raciocínio causal

= 🚀 MELHOR DE AMBOS

EXEMPLOS:
• AlphaGo: Neural (percepção) + MCTS (planejamento)
• Neural Theorem Provers
• Knowledge-Augmented LLMs (RAG)

DESAFIOS:
• Integração eficiente
• Representação compatível
• Escalabilidade
"""

ax4.text(0.1, 0.5, texto_futuro, fontsize=10, family='monospace',
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax4.set_title("O Futuro da IA", fontsize=13, fontweight='bold')

plt.suptitle("Paradigmas de IA: Simbólica vs Conexionista", 
            fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()

# Análise final
print("\n💡 LIÇÕES CHAVE:")
print("="*70)
print("1️⃣ NÃO HÁ 'MELHOR ABSOLUTO' - Depende da tarefa")
print("2️⃣ SIMBÓLICA: Transparente mas frágil")
print("3️⃣ CONEXIONISTA: Robusto mas opaco")
print("4️⃣ FUTURO: Híbrido (Neuro-Simbólico)")
print("5️⃣ XAI (Explainable AI) tenta recuperar interpretabilidade")

print("\n🌐 NEURO-SIMBÓLICO EM AÇÃO:")
print("   • AlphaGo = Neural (avaliação) + MCTS (busca)")
print("   • LLMs + RAG = Neural (linguagem) + Bases de conhecimento")
print("   • Neural Theorem Provers = Neural (heurística) + Simbólica (prova)")

print("\n✅ COMPARAÇÃO COMPLETA!")
