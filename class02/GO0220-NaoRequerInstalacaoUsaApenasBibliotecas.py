# GO0220-NãoRequerInstalaçãoUsaApenasBibliotecas
from typing import Dict, List, Set, Tuple, Any, Optional
import random
from datetime import datetime
import json

# ═══════════════════════════════════════════════════════════════════
# 1. ASSISTENTE VIRTUAL INTELIGENTE - ARQUITETURA
# ═══════════════════════════════════════════════════════════════════

print("="*70)
print("ASSISTENTE VIRTUAL INTELIGENTE - SISTEMA HÍBRIDO")
print("="*70)

class KnowledgeBase:
    """
    Base de Conhecimento Híbrida

    Integra:
    - Fatos (triplas RDF-style)
    - Regras de produção
    - Probabilidades (Bayesianas)
    - Hierarquia de conceitos (DL-style)
    """

    def __init__(self):
        # Knowledge Graph (triplas)
        self.triplas: Set[Tuple[str, str, str]] = set()

        # Hierarquia de conceitos
        self.taxonomia: Dict[str, Set[str]] = {}  # conceito -> superconceitos

        # Regras de inferência
        self.regras: List[Dict] = []

        # Probabilidades (simplificado)
        self.probs: Dict[str, float] = {}

        # Cache de inferências
        self.cache_inferencias: Set[Tuple[str, str, str]] = set()

    def adicionar_tripla(self, sujeito: str, predicado: str, objeto: str):
        """Adicionar fato (tripla RDF-style)"""
        self.triplas.add((sujeito, predicado, objeto))

    def adicionar_taxonomia(self, conceito: str, superconceito: str):
        """Adicionar relação de subsumption"""
        if conceito not in self.taxonomia:
            self.taxonomia[conceito] = set()
        self.taxonomia[conceito].add(superconceito)

    def adicionar_regra(self, nome: str, condicoes: List[Tuple], 
                       conclusoes: List[Tuple]):
        """Adicionar regra de inferência"""
        self.regras.append({
            "nome": nome,
            "condicoes": condicoes,
            "conclusoes": conclusoes
        })

    def adicionar_probabilidade(self, evento: str, prob: float):
        """Adicionar probabilidade a priori"""
        self.probs[evento] = prob

    def consultar(self, sujeito: str = None, predicado: str = None, 
                 objeto: str = None) -> Set[Tuple[str, str, str]]:
        """
        Consultar triplas (com wildcards)

        Exemplo: consultar(sujeito="João", predicado=None, objeto=None)
        Retorna todas as triplas com João como sujeito
        """
        resultado = set()

        for s, p, o in self.triplas:
            if (sujeito is None or s == sujeito) and \
               (predicado is None or p == predicado) and \
               (objeto is None or o == objeto):
                resultado.add((s, p, o))

        return resultado

    def eh_subconceito(self, conceito: str, superconceito: str) -> bool:
        """Verificar subsumption (transitivo)"""
        if conceito == superconceito:
            return True

        if conceito not in self.taxonomia:
            return False

        if superconceito in self.taxonomia[conceito]:
            return True

        # Transitivo
        for super_direto in self.taxonomia[conceito]:
            if self.eh_subconceito(super_direto, superconceito):
                return True

        return False

    def aplicar_regras(self, max_iteracoes: int = 10) -> int:
        """
        Forward chaining: Aplicar regras até quiescência

        Returns:
            Número de novas inferências
        """
        novas_inferencias = 0

        for iteracao in range(max_iteracoes):
            inferencias_ciclo = 0

            for regra in self.regras:
                # Verificar se condições são satisfeitas
                condicoes_satisfeitas = all(
                    tripla in self.triplas for tripla in regra["condicoes"]
                )

                if condicoes_satisfeitas:
                    # Aplicar conclusões
                    for conclusao in regra["conclusoes"]:
                        if conclusao not in self.triplas and \
                           conclusao not in self.cache_inferencias:
                            self.triplas.add(conclusao)
                            self.cache_inferencias.add(conclusao)
                            inferencias_ciclo += 1

            novas_inferencias += inferencias_ciclo

            if inferencias_ciclo == 0:
                break  # Quiescência

        return novas_inferencias

    def explicar(self, tripla: Tuple[str, str, str]) -> List[str]:
        """Explicar como tripla foi derivada"""
        if tripla not in self.triplas:
            return ["Fato não encontrado"]

        if tripla not in self.cache_inferencias:
            return ["Fato declarado diretamente (não inferido)"]

        # Buscar regra que gerou esta inferência
        for regra in self.regras:
            if tripla in regra["conclusoes"]:
                # Verificar se condições estão satisfeitas
                condicoes_satisfeitas = all(
                    cond in self.triplas for cond in regra["condicoes"]
                )

                if condicoes_satisfeitas:
                    explicacao = [
                        f"Inferido pela regra: {regra['nome']}",
                        "Condições satisfeitas:"
                    ]

                    for cond in regra["condicoes"]:
                        explicacao.append(f"   • {cond}")

                    return explicacao

        return ["Inferência encontrada, mas origem desconhecida"]

    def estatisticas(self) -> Dict:
        """Estatísticas da KB"""
        return {
            "triplas_totais": len(self.triplas),
            "inferencias": len(self.cache_inferencias),
            "conceitos_taxonomia": len(self.taxonomia),
            "regras": len(self.regras),
            "probabilidades": len(self.probs)
        }

class AssiatenteVirtual:
    """
    Assistente Virtual com Raciocínio Simbólico
    """

    def __init__(self, nome: str):
        self.nome = nome
        self.kb = KnowledgeBase()
        self.contexto_conversacao: List[str] = []

    def aprender_fato(self, sujeito: str, predicado: str, objeto: str):
        """Aprender novo fato"""
        self.kb.adicionar_tripla(sujeito, predicado, objeto)
        print(f"   {self.nome}: Aprendi que {sujeito} {predicado} {objeto}")

    def responder_consulta(self, pergunta: str) -> str:
        """
        Responder consulta em linguagem natural

        Simplificado: Parsing por palavras-chave
        """
        pergunta_lower = pergunta.lower()

        # Padrão: "quem é o pai de X?"
        if "pai de" in pergunta_lower:
            nome = pergunta_lower.split("pai de")[1].strip().rstrip("?")
            resultados = self.kb.consultar(sujeito=nome, predicado="temPai")

            if resultados:
                pai = list(resultados)[0][2]
                return f"O pai de {nome} é {pai}"
            else:
                return f"Não sei quem é o pai de {nome}"

        # Padrão: "X é um Y?"
        elif " é um " in pergunta_lower or " é uma " in pergunta_lower:
            partes = pergunta_lower.replace("?", "").split(" é um ")
            if len(partes) < 2:
                partes = pergunta_lower.replace("?", "").split(" é uma ")

            if len(partes) == 2:
                individuo = partes[0].strip()
                conceito = partes[1].strip()

                # Verificar tipo
                tipo_triplas = self.kb.consultar(sujeito=individuo, predicado="tipo")

                if tipo_triplas:
                    tipo_direto = list(tipo_triplas)[0][2]

                    if self.kb.eh_subconceito(tipo_direto, conceito):
                        return f"Sim, {individuo} é um(a) {conceito} (porque {tipo_direto} é subconceito de {conceito})"
                    else:
                        return f"Não, {individuo} não é um(a) {conceito}"
                else:
                    return f"Não tenho informações sobre {individuo}"

        # Padrão genérico: consulta por sujeito
        else:
            palavras = pergunta_lower.replace("?", "").split()
            if palavras:
                possivel_sujeito = palavras[0]
                resultados = self.kb.consultar(sujeito=possivel_sujeito)

                if resultados:
                    fatos = [f"{s} {p} {o}" for s, p, o in resultados]
                    return f"Sobre {possivel_sujeito}: " + ", ".join(fatos[:3])

        return "Desculpe, não entendi a pergunta"

    def inferir_conhecimento(self):
        """Executar inferência sobre KB"""
        print(f"\n{self.nome}: Deixe-me pensar... (aplicando regras)")

        inferencias = self.kb.aplicar_regras(max_iteracoes=5)

        if inferencias > 0:
            print(f"{self.nome}: Deduzi {inferencias} novos fatos!")
        else:
            print(f"{self.nome}: Não há novas deduções")

    def explicar_fato(self, sujeito: str, predicado: str, objeto: str):
        """Explicar como um fato foi derivado"""
        tripla = (sujeito, predicado, objeto)
        explicacao = self.kb.explicar(tripla)

        print(f"\n{self.nome}: Explicação de ({sujeito}, {predicado}, {objeto}):")
        for linha in explicacao:
            print(f"   {linha}")

    def resumo_conhecimento(self):
        """Resumir conhecimento atual"""
        stats = self.kb.estatisticas()

        print(f"\n{self.nome}: Meu conhecimento atual:")
        print(f"   • Fatos conhecidos: {stats['triplas_totais']}")
        print(f"   • Fatos inferidos: {stats['inferencias']}")
        print(f"   • Conceitos na taxonomia: {stats['conceitos_taxonomia']}")
        print(f"   • Regras de inferência: {stats['regras']}")

# ═══════════════════════════════════════════════════════════════════
# 2. POPULAR BASE DE CONHECIMENTO
# ═══════════════════════════════════════════════════════════════════

print("\n🤖 INICIALIZANDO ASSISTENTE VIRTUAL...")
print("="*70)

assistente = AssiatenteVirtual("Ada")

print(f"\n👋 {assistente.nome}: Olá! Sou Ada, sua assistente virtual inteligente.")
print(f"   Estou pronta para aprender e raciocinar sobre informações!")

# ═══════════════════════════════════════════════════════════════════
# 3. FASE 1: APRENDIZADO (Fatos + Taxonomia)
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("FASE 1: APRENDIZADO")
print("="*70)

print(f"\n{assistente.nome}: Vou aprender sobre famílias e animais...")

# Fatos sobre família
assistente.aprender_fato("João", "temPai", "José")
assistente.aprender_fato("João", "temMae", "Maria")
assistente.aprender_fato("Ana", "temPai", "João")
assistente.aprender_fato("Ana", "temMae", "Carla")
assistente.aprender_fato("Pedro", "temPai", "João")
assistente.aprender_fato("Pedro", "temMae", "Carla")

# Tipos (instances)
assistente.kb.adicionar_tripla("Rex", "tipo", "Cachorro")
assistente.kb.adicionar_tripla("Mimi", "tipo", "Gato")
assistente.kb.adicionar_tripla("Piu", "tipo", "Passaro")

# Taxonomia (subsumption)
print(f"\n{assistente.nome}: Aprendendo hierarquia de conceitos...")

assistente.kb.adicionar_taxonomia("Cachorro", "Mamifero")
assistente.kb.adicionar_taxonomia("Gato", "Mamifero")
assistente.kb.adicionar_taxonomia("Passaro", "Ave")
assistente.kb.adicionar_taxonomia("Mamifero", "Animal")
assistente.kb.adicionar_taxonomia("Ave", "Animal")
assistente.kb.adicionar_taxonomia("Animal", "SerVivo")

print(f"   ✅ Hierarquia: Cachorro → Mamífero → Animal → SerVivo")

# Regras de inferência
print(f"\n{assistente.nome}: Adicionando regras de raciocínio...")

# Regra 1: Se X temPai Y, então Y temFilho X
assistente.kb.adicionar_regra(
    nome="Pai_implica_Filho",
    condicoes=[("João", "temPai", "José")],
    conclusoes=[("José", "temFilho", "João")]
)

# Regra 2: Se X temPai Y e Y temPai Z, então X temAvo Z
assistente.kb.adicionar_regra(
    nome="Transitividade_Avô",
    condicoes=[
        ("Ana", "temPai", "João"),
        ("João", "temPai", "José")
    ],
    conclusoes=[("Ana", "temAvo", "José")]
)

# Regra 3: Irmãos
assistente.kb.adicionar_regra(
    nome="Mesmo_Pai_Irmãos",
    condicoes=[
        ("Ana", "temPai", "João"),
        ("Pedro", "temPai", "João")
    ],
    conclusoes=[("Ana", "irmao", "Pedro")]
)

print(f"   ✅ {len(assistente.kb.regras)} regras de inferência adicionadas")

assistente.resumo_conhecimento()

# ═══════════════════════════════════════════════════════════════════
# 4. FASE 2: INFERÊNCIA
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("FASE 2: INFERÊNCIA")
print("="*70)

assistente.inferir_conhecimento()

assistente.resumo_conhecimento()

# ═══════════════════════════════════════════════════════════════════
# 5. FASE 3: CONSULTAS E EXPLICAÇÕES
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("FASE 3: CONSULTAS INTERATIVAS")
print("="*70)

perguntas = [
    "Quem é o pai de Ana?",
    "Rex é um Animal?",
    "Mimi é um Mamifero?",
    "Piu é um Mamifero?",
]

for pergunta in perguntas:
    print(f"\n👤 Usuário: {pergunta}")
    resposta = assistente.responder_consulta(pergunta)
    print(f"🤖 {assistente.nome}: {resposta}")

# Explicações
print("\n" + "="*70)
print("FASE 4: EXPLICAÇÕES")
print("="*70)

print(f"\n👤 Usuário: Como você sabe que José é pai de João?")
assistente.explicar_fato("José", "temFilho", "João")

print(f"\n👤 Usuário: Como você sabe que Ana e Pedro são irmãos?")
assistente.explicar_fato("Ana", "irmao", "Pedro")

# ═══════════════════════════════════════════════════════════════════
# 6. FASE 5: RACIOCÍNIO PROBABILÍSTICO
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("FASE 5: RACIOCÍNIO PROBABILÍSTICO (Simplificado)")
print("="*70)

# Adicionar probabilidades
assistente.kb.adicionar_probabilidade("chuva_amanha", 0.3)
assistente.kb.adicionar_probabilidade("transito_ruim_se_chuva", 0.7)

print(f"\n{assistente.nome}: Aprendi sobre probabilidades:")
print(f"   • P(chuva amanhã) = 0.3")
print(f"   • P(trânsito ruim | chuva) = 0.7")

# Inferência probabilística (simplificado)
prob_chuva = assistente.kb.probs["chuva_amanha"]
prob_transito_chuva = assistente.kb.probs["transito_ruim_se_chuva"]
prob_transito_e_chuva = prob_chuva * prob_transito_chuva

print(f"\n👤 Usuário: Qual a probabilidade de trânsito ruim amanhã?")
print(f"🤖 {assistente.nome}: Considerando apenas a chuva:")
print(f"   P(trânsito ruim E chuva) = {prob_transito_e_chuva:.2f}")
print(f"   (Isso é uma simplificação - na realidade, usaria redes Bayesianas)")

# ═══════════════════════════════════════════════════════════════════
# 7. FASE 6: DEMONSTRAÇÃO DE INTEGRAÇÃO
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("FASE 6: DEMONSTRAÇÃO DE INTEGRAÇÃO")
print("="*70)

print(f"\n{assistente.nome}: Este sistema demonstra integração de:")
print(f"\n   1️⃣ KNOWLEDGE GRAPH (Triplas RDF)")
print(f"      • Fatos: (João, temPai, José), (Rex, tipo, Cachorro)")
print(f"      • Consultas: Busca por sujeito/predicado/objeto")
print(f"\n   2️⃣ DESCRIPTION LOGIC (Taxonomia)")
print(f"      • Hierarquia: Cachorro ⊑ Mamífero ⊑ Animal ⊑ SerVivo")
print(f"      • Subsumption: Rex é Animal? → Sim (via taxonomia)")
print(f"\n   3️⃣ REGRAS DE PRODUÇÃO (Forward Chaining)")
print(f"      • Regras: IF X temPai Y THEN Y temFilho X")
print(f"      • Inferência: José temFilho João (derivado)")
print(f"\n   4️⃣ RACIOCÍNIO PROBABILÍSTICO (Bayesiano)")
print(f"      • Probabilidades: P(chuva) = 0.3")
print(f"      • Inferência: P(trânsito ruim | chuva) = 0.7")
print(f"\n   5️⃣ EXPLICABILIDADE")
print(f"      • Trace de inferências: Como fato foi derivado")
print(f"      • Regras aplicadas: Qual regra gerou conclusão")

# ═══════════════════════════════════════════════════════════════════
# 8. VISUALIZAÇÃO FINAL
# ═══════════════════════════════════════════════════════════════════

print("\n📊 GERANDO VISUALIZAÇÃO INTEGRADA...")

import matplotlib.pyplot as plt
import networkx as nx

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 1. Knowledge Graph
ax1 = axes[0, 0]
G_kg = nx.DiGraph()

# Adicionar arestas (triplas)
for s, p, o in list(assistente.kb.triplas)[:15]:  # Limitar para visualização
    G_kg.add_edge(s, o, label=p)

pos1 = nx.spring_layout(G_kg, k=2, iterations=50, seed=42)

# Cores: pessoas vs animais
cores1 = []
for no in G_kg.nodes():
    if no in ["João", "José", "Maria", "Ana", "Pedro", "Carla"]:
        cores1.append('lightblue')
    elif no in ["Rex", "Mimi", "Piu"]:
        cores1.append('lightcoral')
    else:
        cores1.append('lightgreen')

nx.draw_networkx_nodes(G_kg, pos1, node_color=cores1, node_size=1500, ax=ax1)
nx.draw_networkx_labels(G_kg, pos1, font_size=8, ax=ax1)
nx.draw_networkx_edges(G_kg, pos1, edge_color='gray', arrows=True, 
                       arrowsize=15, ax=ax1)

# Rótulos de arestas (simplificado)
edge_labels = {(u, v): d['label'][:8] for u, v, d in G_kg.edges(data=True)}
nx.draw_networkx_edge_labels(G_kg, pos1, edge_labels, font_size=6, ax=ax1)

ax1.set_title("Knowledge Graph (Triplas)", fontsize=12, fontweight='bold')
ax1.axis('off')

# 2. Taxonomia
ax2 = axes[0, 1]
G_tax = nx.DiGraph()

for conceito, superconceitos in assistente.kb.taxonomia.items():
    for super_c in superconceitos:
        G_tax.add_edge(conceito, super_c)

pos2 = nx.spring_layout(G_tax, k=2.5, iterations=50, seed=42)

nx.draw_networkx_nodes(G_tax, pos2, node_color='lightyellow', 
                      node_size=2000, ax=ax2)
nx.draw_networkx_labels(G_tax, pos2, font_size=9, ax=ax2)
nx.draw_networkx_edges(G_tax, pos2, edge_color='orange', arrows=True, 
                      arrowsize=20, width=2, ax=ax2)

ax2.set_title("Taxonomia (Subsumption)", fontsize=12, fontweight='bold')
ax2.axis('off')

# 3. Regras de Inferência
ax3 = axes[1, 0]
ax3.axis('off')

regras_texto = []
for i, regra in enumerate(assistente.kb.regras, 1):
    nome = regra['nome']
    num_cond = len(regra['condicoes'])
    num_conc = len(regra['conclusoes'])
    regras_texto.append(f"{i}. {nome}\n   ({num_cond} condições → {num_conc} conclusões)")

texto_regras = "\n\n".join(regras_texto)
ax3.text(0.1, 0.5, f"REGRAS DE INFERÊNCIA:\n\n{texto_regras}", 
        fontsize=10, verticalalignment='center', 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax3.set_title("Regras de Produção", fontsize=12, fontweight='bold')

# 4. Estatísticas
ax4 = axes[1, 1]
ax4.axis('off')

stats = assistente.kb.estatisticas()

stats_texto = f"""ESTATÍSTICAS DA BASE DE CONHECIMENTO:

📊 Dados:
   • Triplas totais: {stats['triplas_totais']}
   • Fatos inferidos: {stats['inferencias']}
   • Fatos declarados: {stats['triplas_totais'] - stats['inferencias']}

🧠 Estruturas:
   • Conceitos na taxonomia: {stats['conceitos_taxonomia']}
   • Regras de inferência: {stats['regras']}
   • Probabilidades: {stats['probabilidades']}

🎯 Capacidades:
   ✅ Consultas em triplas
   ✅ Inferência via regras
   ✅ Subsumption hierárquica
   ✅ Explicação de fatos
   ✅ Raciocínio probabilístico
"""

ax4.text(0.1, 0.5, stats_texto, fontsize=9, verticalalignment='center',
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax4.set_title("Estatísticas e Capacidades", fontsize=12, fontweight='bold')

plt.suptitle("Sistema Híbrido de IA Simbólica - Integração Completa", 
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════
# 9. RELATÓRIO FINAL
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("RELATÓRIO FINAL - SISTEMA HÍBRIDO")
print("="*70)

print(f"\n🏗️ ARQUITETURA INTEGRADA:")
print(f"   1. Knowledge Graph → Armazenamento de fatos (triplas)")
print(f"   2. Description Logic → Hierarquia de conceitos")
print(f"   3. Production Rules → Inferência forward chaining")
print(f"   4. Bayesian Reasoning → Incerteza e probabilidades")
print(f"   5. Explanation → Trace de raciocínio")

print(f"\n🎯 VANTAGENS DA INTEGRAÇÃO:")
print(f"   ✅ FLEXIBILIDADE: Diferentes representações para diferentes problemas")
print(f"   ✅ RIGOR: Lógica formal + Probabilidades")
print(f"   ✅ ESCALABILIDADE: Knowledge graphs podem crescer")
print(f"   ✅ EXPLICABILIDADE: Trace completo de inferências")
print(f"   ✅ INTEROPERABILIDADE: Padrões (RDF, OWL, SPARQL)")

print(f"\n💡 IA SIMBÓLICA vs CONEXIONISTA:")
print(f"\n   SIMBÓLICA (Este sistema):")
print(f"      ✅ Explicável (trace de raciocínio)")
print(f"      ✅ Rigorosa (lógica formal)")
print(f"      ✅ Composicional (regras modulares)")
print(f"      ❌ Requer conhecimento explícito")
print(f"      ❌ Frágil (não generaliza bem)")
print(f"\n   CONEXIONISTA (Redes Neurais):")
print(f"      ✅ Aprende de dados automaticamente")
print(f"      ✅ Generaliza (padrões estatísticos)")
print(f"      ✅ Robusto a ruído")
print(f"      ❌ Black-box (difícil explicar)")
print(f"      ❌ Requer muitos dados")

print(f"\n🚀 FUTURO: IA NEURO-SIMBÓLICA:")
print(f"   • Combina Simbólico + Neural")
print(f"   • Exemplos: AlphaGo (MCTS + Deep Learning)")
print(f"   • Promessa: Raciocínio + Aprendizado")
print(f"   • Desafio: Integração eficiente")

print(f"\n🎓 LIÇÕES APRENDIDAS NESTA AULA:")
print(f"   1. Múltiplas formas de representar conhecimento")
print(f"   2. Cada técnica tem pontos fortes")
print(f"   3. Integração amplifica capacidades")
print(f"   4. IA Simbólica complementa (não substitui) ML")
print(f"   5. Explicabilidade é crucial em aplicações críticas")

print("\n✅ PROJETO INTEGRADOR COMPLETO!")
print(f"\n{assistente.nome}: Foi um prazer demonstrar minhas capacidades!")
print(f"   Até a próxima aula! 👋")
