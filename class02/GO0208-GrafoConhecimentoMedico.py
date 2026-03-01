# GO0208-GrafoConhecimentoMédico
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# ═══════════════════════════════════════════════════════════════════
# 1. GRAFO DE CONHECIMENTO MÉDICO
# ═══════════════════════════════════════════════════════════════════

print("="*70)
print("GRAFO DE CONHECIMENTO MÉDICO - REPRESENTAÇÃO SIMBÓLICA")
print("="*70)

class KnowledgeGraph:
    """
    Grafo de conhecimento usando triplas RDF-style
    (sujeito, predicado, objeto)
    """

    def __init__(self):
        self.triplas = []
        self.graph = nx.DiGraph()

    def adicionar_tripla(self, sujeito, predicado, objeto):
        """Adicionar tripla ao grafo"""
        self.triplas.append((sujeito, predicado, objeto))
        self.graph.add_edge(sujeito, objeto, relacao=predicado)

    def buscar_por_sujeito(self, sujeito):
        """Encontrar todas as triplas com sujeito específico"""
        return [(s, p, o) for s, p, o in self.triplas if s == sujeito]

    def buscar_por_predicado(self, predicado):
        """Encontrar todas as triplas com predicado específico"""
        return [(s, p, o) for s, p, o in self.triplas if p == predicado]

    def buscar_por_objeto(self, objeto):
        """Encontrar todas as triplas com objeto específico"""
        return [(s, p, o) for s, p, o in self.triplas if o == objeto]

    def inferir_transitividade(self, relacao):
        """
        Inferir relações transitivas
        Ex: se A é_tipo B e B é_tipo C, então A é_tipo C
        """
        novas_triplas = []

        for s1, p1, o1 in self.triplas:
            if p1 == relacao:
                for s2, p2, o2 in self.triplas:
                    if p2 == relacao and s2 == o1:
                        # Transitividade: s1 -> o1 -> o2
                        nova = (s1, relacao, o2)
                        if nova not in self.triplas and nova not in novas_triplas:
                            novas_triplas.append(nova)

        return novas_triplas

    def inferir_simetria(self, relacao):
        """
        Inferir relações simétricas
        Ex: se A relacionado_com B, então B relacionado_com A
        """
        novas_triplas = []

        for s, p, o in self.triplas:
            if p == relacao:
                nova = (o, relacao, s)
                if nova not in self.triplas and nova not in novas_triplas:
                    novas_triplas.append(nova)

        return novas_triplas

    def diagnosticar(self, sintomas):
        """
        Diagnosticar possíveis doenças baseado em sintomas
        """
        print(f"\n🔍 DIAGNÓSTICO para sintomas: {sintomas}")

        # Encontrar doenças que causam esses sintomas
        doencas_possiveis = defaultdict(int)

        for sintoma in sintomas:
            # Buscar doenças que causam este sintoma
            triplas_sintoma = self.buscar_por_objeto(sintoma)

            for s, p, o in triplas_sintoma:
                if p == "causa_sintoma":
                    doencas_possiveis[s] += 1

        # Ordenar por match de sintomas
        ranking = sorted(doencas_possiveis.items(), key=lambda x: x[1], reverse=True)

        print("\n📊 DOENÇAS POSSÍVEIS:")
        for doenca, matches in ranking:
            confianca = (matches / len(sintomas)) * 100
            print(f"   • {doenca}: {matches}/{len(sintomas)} sintomas ({confianca:.1f}%)")

            # Mostrar tratamentos
            tratamentos = self.buscar_por_sujeito(doenca)
            for s, p, o in tratamentos:
                if p == "tratado_com":
                    print(f"      → Tratamento: {o}")

        return ranking

    def visualizar(self, filename='grafo_conhecimento.png'):
        """Visualizar o grafo"""
        plt.figure(figsize=(16, 12))

        # Layout
        pos = nx.spring_layout(self.graph, k=2, iterations=50)

        # Cores por tipo de nó
        node_colors = []
        for node in self.graph.nodes():
            if any(p == "tipo_de" and o == "Doença" for s, p, o in self.triplas if s == node):
                node_colors.append('lightcoral')
            elif any(p == "tipo_de" and o == "Sintoma" for s, p, o in self.triplas if s == node):
                node_colors.append('lightblue')
            elif any(p == "tipo_de" and o == "Tratamento" for s, p, o in self.triplas if s == node):
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightgray')

        # Desenhar nós
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, 
                              node_size=3000, alpha=0.9)

        # Desenhar arestas
        nx.draw_networkx_edges(self.graph, pos, arrows=True, 
                              arrowsize=20, alpha=0.6, width=2)

        # Labels dos nós
        nx.draw_networkx_labels(self.graph, pos, font_size=9, font_weight='bold')

        # Labels das arestas (relações)
        edge_labels = {(u, v): d['relacao'] 
                      for u, v, d in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, 
                                    font_size=7, font_color='red')

        plt.title("Grafo de Conhecimento Médico", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# Criar grafo de conhecimento
kg = KnowledgeGraph()

print("\n📚 CONSTRUINDO BASE DE CONHECIMENTO MÉDICA...")

# ═══════════════════════════════════════════════════════════════════
# 2. POPULAR GRAFO COM CONHECIMENTO MÉDICO
# ═══════════════════════════════════════════════════════════════════

# Doenças
kg.adicionar_tripla("Gripe", "tipo_de", "Doença")
kg.adicionar_tripla("COVID-19", "tipo_de", "Doença")
kg.adicionar_tripla("Pneumonia", "tipo_de", "Doença")
kg.adicionar_tripla("Dengue", "tipo_de", "Doença")
kg.adicionar_tripla("Alergia", "tipo_de", "Doença")

# Sintomas
kg.adicionar_tripla("Febre", "tipo_de", "Sintoma")
kg.adicionar_tripla("Tosse", "tipo_de", "Sintoma")
kg.adicionar_tripla("Dor_Cabeça", "tipo_de", "Sintoma")
kg.adicionar_tripla("Cansaço", "tipo_de", "Sintoma")
kg.adicionar_tripla("Dor_Corpo", "tipo_de", "Sintoma")
kg.adicionar_tripla("Falta_Ar", "tipo_de", "Sintoma")
kg.adicionar_tripla("Coriza", "tipo_de", "Sintoma")
kg.adicionar_tripla("Náusea", "tipo_de", "Sintoma")

# Tratamentos
kg.adicionar_tripla("Paracetamol", "tipo_de", "Tratamento")
kg.adicionar_tripla("Repouso", "tipo_de", "Tratamento")
kg.adicionar_tripla("Hidratação", "tipo_de", "Tratamento")
kg.adicionar_tripla("Antibiótico", "tipo_de", "Tratamento")
kg.adicionar_tripla("Antialérgico", "tipo_de", "Tratamento")
kg.adicionar_tripla("Isolamento", "tipo_de", "Tratamento")

# Relações: Doença -> causa_sintoma -> Sintoma
# Gripe
kg.adicionar_tripla("Gripe", "causa_sintoma", "Febre")
kg.adicionar_tripla("Gripe", "causa_sintoma", "Tosse")
kg.adicionar_tripla("Gripe", "causa_sintoma", "Dor_Cabeça")
kg.adicionar_tripla("Gripe", "causa_sintoma", "Cansaço")
kg.adicionar_tripla("Gripe", "causa_sintoma", "Dor_Corpo")

# COVID-19
kg.adicionar_tripla("COVID-19", "causa_sintoma", "Febre")
kg.adicionar_tripla("COVID-19", "causa_sintoma", "Tosse")
kg.adicionar_tripla("COVID-19", "causa_sintoma", "Cansaço")
kg.adicionar_tripla("COVID-19", "causa_sintoma", "Falta_Ar")
kg.adicionar_tripla("COVID-19", "causa_sintoma", "Dor_Corpo")

# Pneumonia
kg.adicionar_tripla("Pneumonia", "causa_sintoma", "Febre")
kg.adicionar_tripla("Pneumonia", "causa_sintoma", "Tosse")
kg.adicionar_tripla("Pneumonia", "causa_sintoma", "Falta_Ar")
kg.adicionar_tripla("Pneumonia", "causa_sintoma", "Cansaço")

# Dengue
kg.adicionar_tripla("Dengue", "causa_sintoma", "Febre")
kg.adicionar_tripla("Dengue", "causa_sintoma", "Dor_Cabeça")
kg.adicionar_tripla("Dengue", "causa_sintoma", "Dor_Corpo")
kg.adicionar_tripla("Dengue", "causa_sintoma", "Náusea")

# Alergia
kg.adicionar_tripla("Alergia", "causa_sintoma", "Coriza")
kg.adicionar_tripla("Alergia", "causa_sintoma", "Tosse")

# Relações: Doença -> tratado_com -> Tratamento
kg.adicionar_tripla("Gripe", "tratado_com", "Paracetamol")
kg.adicionar_tripla("Gripe", "tratado_com", "Repouso")
kg.adicionar_tripla("Gripe", "tratado_com", "Hidratação")

kg.adicionar_tripla("COVID-19", "tratado_com", "Isolamento")
kg.adicionar_tripla("COVID-19", "tratado_com", "Repouso")
kg.adicionar_tripla("COVID-19", "tratado_com", "Hidratação")

kg.adicionar_tripla("Pneumonia", "tratado_com", "Antibiótico")
kg.adicionar_tripla("Pneumonia", "tratado_com", "Repouso")

kg.adicionar_tripla("Dengue", "tratado_com", "Hidratação")
kg.adicionar_tripla("Dengue", "tratado_com", "Paracetamol")
kg.adicionar_tripla("Dengue", "tratado_com", "Repouso")

kg.adicionar_tripla("Alergia", "tratado_com", "Antialérgico")

print(f"✅ Base construída: {len(kg.triplas)} triplas")

# ═══════════════════════════════════════════════════════════════════
# 3. CONSULTAS E INFERÊNCIAS
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("CONSULTAS AO GRAFO DE CONHECIMENTO")
print("="*70)

# Consulta 1: Sintomas da Gripe
print("\n🔍 Consulta 1: Quais sintomas a Gripe causa?")
sintomas_gripe = kg.buscar_por_sujeito("Gripe")
for s, p, o in sintomas_gripe:
    if p == "causa_sintoma":
        print(f"   • {o}")

# Consulta 2: Doenças que causam Febre
print("\n🔍 Consulta 2: Quais doenças causam Febre?")
doencas_febre = kg.buscar_por_objeto("Febre")
for s, p, o in doencas_febre:
    if p == "causa_sintoma":
        print(f"   • {s}")

# Consulta 3: Tratamentos
print("\n🔍 Consulta 3: Todos os tratamentos disponíveis:")
tratamentos = kg.buscar_por_predicado("tratado_com")
tratamentos_unicos = set([o for s, p, o in tratamentos])
for t in tratamentos_unicos:
    print(f"   • {t}")

# ═══════════════════════════════════════════════════════════════════
# 4. DIAGNÓSTICO BASEADO EM SINTOMAS
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("SISTEMA DE DIAGNÓSTICO")
print("="*70)

# Cenário 1: Paciente com gripe
caso1 = ["Febre", "Tosse", "Dor_Cabeça", "Cansaço"]
kg.diagnosticar(caso1)

# Cenário 2: Paciente com COVID
caso2 = ["Febre", "Tosse", "Falta_Ar", "Cansaço"]
kg.diagnosticar(caso2)

# Cenário 3: Paciente com dengue
caso3 = ["Febre", "Dor_Cabeça", "Dor_Corpo", "Náusea"]
kg.diagnosticar(caso3)

# ═══════════════════════════════════════════════════════════════════
# 5. INFERÊNCIAS LÓGICAS
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("INFERÊNCIAS LÓGICAS")
print("="*70)

# Adicionar hierarquia de doenças
kg.adicionar_tripla("Gripe", "é_tipo_de", "Doença_Viral")
kg.adicionar_tripla("COVID-19", "é_tipo_de", "Doença_Viral")
kg.adicionar_tripla("Doença_Viral", "é_tipo_de", "Doença_Infecciosa")
kg.adicionar_tripla("Pneumonia", "é_tipo_de", "Doença_Respiratória")

# Inferir transitividade
print("\n🧠 INFERÊNCIA TRANSITIVA (é_tipo_de):")
print("   Se A é_tipo_de B e B é_tipo_de C, então A é_tipo_de C")

novas_transitivas = kg.inferir_transitividade("é_tipo_de")
for s, p, o in novas_transitivas:
    print(f"   ✨ INFERIDO: {s} {p} {o}")
    kg.adicionar_tripla(s, p, o)

# ═══════════════════════════════════════════════════════════════════
# 6. ESTATÍSTICAS DO GRAFO
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("ESTATÍSTICAS DO GRAFO DE CONHECIMENTO")
print("="*70)

print(f"\n📊 MÉTRICAS:")
print(f"   Total de triplas: {len(kg.triplas)}")
print(f"   Total de nós: {kg.graph.number_of_nodes()}")
print(f"   Total de arestas: {kg.graph.number_of_edges()}")

# Contar por tipo de relação
relacoes = defaultdict(int)
for s, p, o in kg.triplas:
    relacoes[p] += 1

print(f"\n📈 RELAÇÕES:")
for rel, count in sorted(relacoes.items(), key=lambda x: x[1], reverse=True):
    print(f"   • {rel}: {count}")

# Nós mais conectados
print(f"\n🔗 NÓS MAIS CONECTADOS:")
degree_dict = dict(kg.graph.degree())
top_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:5]
for node, degree in top_nodes:
    print(f"   • {node}: {degree} conexões")

# ═══════════════════════════════════════════════════════════════════
# 7. VISUALIZAÇÃO
# ═══════════════════════════════════════════════════════════════════

print("\n📊 Gerando visualização do grafo...")
kg.visualizar()

# ═══════════════════════════════════════════════════════════════════
# 8. RELATÓRIO FINAL
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("RELATÓRIO FINAL - GRAFO DE CONHECIMENTO MÉDICO")
print("="*70)

print(f"\n🏗️ ESTRUTURA:")
print(f"   • Modelo: RDF triplas (sujeito-predicado-objeto)")
print(f"   • Triplas: {len(kg.triplas)}")
print(f"   • Nós: {kg.graph.number_of_nodes()}")
print(f"   • Entidades: Doenças, Sintomas, Tratamentos")

print(f"\n🔍 FUNCIONALIDADES:")
print(f"   ✅ Consultas por sujeito, predicado, objeto")
print(f"   ✅ Diagnóstico baseado em sintomas")
print(f"   ✅ Inferência transitiva")
print(f"   ✅ Recomendação de tratamentos")

print(f"\n🎯 APLICAÇÕES REAIS:")
print(f"   • Sistemas especialistas médicos")
print(f"   • Chatbots de triagem")
print(f"   • Base de conhecimento para IA médica")
print(f"   • Suporte a decisão clínica")

print(f"\n💡 VANTAGENS DA REPRESENTAÇÃO SIMBÓLICA:")
print(f"   ✅ EXPLICÁVEL: Raciocínio transparente")
print(f"   ✅ EXTENSÍVEL: Fácil adicionar conhecimento")
print(f"   ✅ LÓGICO: Inferências formais")
print(f"   ✅ ESTRUTURADO: Relações explícitas")

print("\n✅ GRAFO DE CONHECIMENTO MÉDICO COMPLETO!")
