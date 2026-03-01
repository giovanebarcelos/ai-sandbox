# GO0219-NãoRequerInstalaçãoUsaApenasBibliotecas
from typing import Set, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════
# 1. DESCRIPTION LOGIC - FUNDAMENTOS
# ═══════════════════════════════════════════════════════════════════

print("="*70)
print("DESCRIPTION LOGIC - CLASSIFICAÇÃO DE ESPÉCIES")
print("="*70)

@dataclass
class Conceito:
    """
    Conceito (Classe) em Description Logic

    Exemplos: Animal, Mamífero, Carnívoro
    """
    nome: str
    superclasses: Set[str] = field(default_factory=set)
    propriedades_necessarias: Dict[str, str] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.nome)

    def __repr__(self):
        return f"Conceito({self.nome})"

@dataclass
class Individuo:
    """
    Indivíduo (Instância) em Description Logic

    Exemplos: rex (um cachorro específico)
    """
    nome: str
    conceitos: Set[str] = field(default_factory=set)  # Classes que pertence
    propriedades: Dict[str, any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.nome)

    def __repr__(self):
        return f"Individuo({self.nome})"

class BaseConhecimentoDL:
    """
    Base de Conhecimento em Description Logic

    Componentes:
    - TBox (Terminologia): Definições de conceitos
    - ABox (Assertions): Fatos sobre indivíduos
    """

    def __init__(self):
        # TBox
        self.conceitos: Dict[str, Conceito] = {}
        self.relacoes: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)

        # ABox
        self.individuos: Dict[str, Individuo] = {}

        # Cache de inferências
        self.cache_subsumption: Dict[Tuple[str, str], bool] = {}

    def adicionar_conceito(self, nome: str, superclasses: List[str] = None, 
                          propriedades: Dict[str, str] = None):
        """Adicionar conceito ao TBox"""
        conceito = Conceito(
            nome=nome,
            superclasses=set(superclasses or []),
            propriedades_necessarias=propriedades or {}
        )
        self.conceitos[nome] = conceito

    def adicionar_individuo(self, nome: str, conceitos: List[str], 
                           propriedades: Dict[str, any] = None):
        """Adicionar indivíduo ao ABox"""
        individuo = Individuo(
            nome=nome,
            conceitos=set(conceitos),
            propriedades=propriedades or {}
        )
        self.individuos[nome] = individuo

    def adicionar_relacao(self, relacao: str, origem: str, destino: str):
        """Adicionar relação entre indivíduos"""
        self.relacoes[relacao].add((origem, destino))

    def subsume(self, conceito_geral: str, conceito_especifico: str) -> bool:
        """
        Reasoning Service: Subsumption

        Verifica se conceito_especifico é subconceitoده de conceito_geral
        (conceito_especifico ⊑ conceito_geral)
        """
        # Cache
        chave = (conceito_geral, conceito_especifico)
        if chave in self.cache_subsumption:
            return self.cache_subsumption[chave]

        # Caso trivial
        if conceito_geral == conceito_especifico:
            self.cache_subsumption[chave] = True
            return True

        # Verificar se conceito_especifico tem conceito_geral como superclasse
        if conceito_especifico not in self.conceitos:
            self.cache_subsumption[chave] = False
            return False

        conceito_esp = self.conceitos[conceito_especifico]

        # Subsumption direta
        if conceito_geral in conceito_esp.superclasses:
            self.cache_subsumption[chave] = True
            return True

        # Subsumption transitiva (via superclasses)
        for superclasse in conceito_esp.superclasses:
            if self.subsume(conceito_geral, superclasse):
                self.cache_subsumption[chave] = True
                return True

        self.cache_subsumption[chave] = False
        return False

    def classificar_individuo(self, nome_individuo: str) -> Set[str]:
        """
        Reasoning Service: Classification

        Inferir todos os conceitos aos quais o indivíduo pertence
        """
        if nome_individuo not in self.individuos:
            return set()

        individuo = self.individuos[nome_individuo]
        conceitos_inferidos = set(individuo.conceitos)

        # Para cada conceito direto, adicionar superclasses
        for conceito_direto in list(individuo.conceitos):
            for conceito_geral in self.conceitos.keys():
                if self.subsume(conceito_geral, conceito_direto):
                    conceitos_inferidos.add(conceito_geral)

        # Verificar propriedades necessárias
        for conceito_nome, conceito in self.conceitos.items():
            if conceito_nome in conceitos_inferidos:
                continue

            # Verificar se indivíduo satisfaz propriedades necessárias
            satisfaz = True
            for prop, valor_req in conceito.propriedades_necessarias.items():
                if prop not in individuo.propriedades:
                    satisfaz = False
                    break

                if individuo.propriedades[prop] != valor_req:
                    satisfaz = False
                    break

            if satisfaz and conceito.propriedades_necessarias:
                conceitos_inferidos.add(conceito_nome)

        return conceitos_inferidos

    def consultar_individuos(self, conceito: str) -> List[str]:
        """
        Consultar indivíduos que pertencem a conceito

        Args:
            conceito: Nome do conceito

        Returns:
            Lista de nomes de indivíduos
        """
        resultado = []

        for nome_ind, individuo in self.individuos.items():
            conceitos_completos = self.classificar_individuo(nome_ind)
            if conceito in conceitos_completos:
                resultado.append(nome_ind)

        return resultado

    def explicar_classificacao(self, nome_individuo: str):
        """Explicar por que indivíduo pertence a conceitos"""
        if nome_individuo not in self.individuos:
            print(f"   ❌ Indivíduo '{nome_individuo}' não encontrado")
            return

        individuo = self.individuos[nome_individuo]
        conceitos_todos = self.classificar_individuo(nome_individuo)
        conceitos_diretos = individuo.conceitos
        conceitos_inferidos = conceitos_todos - conceitos_diretos

        print(f"\n🔍 EXPLICAÇÃO: {nome_individuo}")
        print("-"*70)

        print(f"\n   Conceitos diretos (declarados):")
        for conceito in sorted(conceitos_diretos):
            print(f"      • {conceito}")

        if conceitos_inferidos:
            print(f"\n   Conceitos inferidos (via subsumption):")
            for conceito in sorted(conceitos_inferidos):
                # Encontrar caminho de inferência
                for conceito_direto in conceitos_diretos:
                    if self.subsume(conceito, conceito_direto):
                        print(f"      • {conceito} (porque {conceito_direto} ⊑ {conceito})")
                        break

        print(f"\n   Propriedades:")
        for prop, valor in sorted(individuo.propriedades.items()):
            print(f"      • {prop} = {valor}")

    def resumo(self):
        """Resumo da base de conhecimento"""
        return {
            "conceitos": len(self.conceitos),
            "individuos": len(self.individuos),
            "relacoes": len(self.relacoes)
        }

# ═══════════════════════════════════════════════════════════════════
# 2. CONSTRUIR ONTOLOGIA DE ESPÉCIES
# ═══════════════════════════════════════════════════════════════════

print("\n🌍 CONSTRUINDO ONTOLOGIA DE ESPÉCIES...")
print("="*70)

kb = BaseConhecimentoDL()

# TBox (Terminologia) - Hierarquia de classes

# Nível 1: Ser Vivo
kb.adicionar_conceito("SerVivo")

# Nível 2: Animal, Planta
kb.adicionar_conceito("Animal", superclasses=["SerVivo"])
kb.adicionar_conceito("Planta", superclasses=["SerVivo"])

# Nível 3: Vertebrado, Invertebrado
kb.adicionar_conceito("Vertebrado", superclasses=["Animal"])
kb.adicionar_conceito("Invertebrado", superclasses=["Animal"])

# Nível 4: Mamífero, Ave, Réptil
kb.adicionar_conceito("Mamifero", superclasses=["Vertebrado"], 
                     propriedades={"temperatura": "quente", "pelos": "sim"})
kb.adicionar_conceito("Ave", superclasses=["Vertebrado"],
                     propriedades={"temperatura": "quente", "penas": "sim"})
kb.adicionar_conceito("Reptil", superclasses=["Vertebrado"],
                     propriedades={"temperatura": "fria", "escamas": "sim"})

# Nível 5: Carnívoro, Herbívoro, Onívoro (baseado em dieta)
kb.adicionar_conceito("Carnivoro", superclasses=["Animal"],
                     propriedades={"dieta": "carne"})
kb.adicionar_conceito("Herbivoro", superclasses=["Animal"],
                     propriedades={"dieta": "plantas"})
kb.adicionar_conceito("Onivoro", superclasses=["Animal"],
                     propriedades={"dieta": "mista"})

# Nível 6: Espécies específicas
kb.adicionar_conceito("Cachorro", superclasses=["Mamifero", "Carnivoro"])
kb.adicionar_conceito("Gato", superclasses=["Mamifero", "Carnivoro"])
kb.adicionar_conceito("Vaca", superclasses=["Mamifero", "Herbivoro"])
kb.adicionar_conceito("Aguia", superclasses=["Ave", "Carnivoro"])
kb.adicionar_conceito("Galinha", superclasses=["Ave", "Onivoro"])

print(f"✅ TBox construído: {len(kb.conceitos)} conceitos")

# ABox (Assertions) - Indivíduos

kb.adicionar_individuo("rex", ["Cachorro"], 
                      {"temperatura": "quente", "pelos": "sim", "dieta": "carne"})

kb.adicionar_individuo("mimi", ["Gato"],
                      {"temperatura": "quente", "pelos": "sim", "dieta": "carne"})

kb.adicionar_individuo("mimosa", ["Vaca"],
                      {"temperatura": "quente", "pelos": "sim", "dieta": "plantas"})

kb.adicionar_individuo("zeus", ["Aguia"],
                      {"temperatura": "quente", "penas": "sim", "dieta": "carne"})

kb.adicionar_individuo("chica", ["Galinha"],
                      {"temperatura": "quente", "penas": "sim", "dieta": "mista"})

print(f"✅ ABox construído: {len(kb.individuos)} indivíduos")

resumo = kb.resumo()
print(f"\n📊 BASE DE CONHECIMENTO:")
print(f"   Conceitos: {resumo['conceitos']}")
print(f"   Indivíduos: {resumo['individuos']}")

# ═══════════════════════════════════════════════════════════════════
# 3. REASONING SERVICES
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("REASONING SERVICES")
print("="*70)

# 1. Subsumption
print("\n🔍 SUBSUMPTION (Hierarquia de Conceitos):")
print("-"*70)

pares_teste = [
    ("Animal", "Cachorro"),
    ("Vertebrado", "Mamifero"),
    ("SerVivo", "Ave"),
    ("Carnivoro", "Aguia"),
]

for conceito_geral, conceito_esp in pares_teste:
    resultado = kb.subsume(conceito_geral, conceito_esp)
    simbolo = "✅" if resultado else "❌"
    print(f"   {simbolo} {conceito_esp} ⊑ {conceito_geral} ? {resultado}")

# 2. Classification
print("\n🔍 CLASSIFICATION (Classificar Indivíduos):")
print("-"*70)

for nome_ind in kb.individuos.keys():
    conceitos = kb.classificar_individuo(nome_ind)
    print(f"\n   {nome_ind} ∈ {sorted(conceitos)}")

# 3. Explicação
kb.explicar_classificacao("rex")
kb.explicar_classificacao("zeus")

# 4. Consulta: Quais animais são carnívoros?
print("\n🔍 CONSULTA: Quais animais são Carnívoros?")
print("-"*70)

carnivoros = kb.consultar_individuos("Carnivoro")
print(f"   Carnívoros: {carnivoros}")

# 5. Consulta: Quais são mamíferos?
print("\n🔍 CONSULTA: Quais são Mamíferos?")
print("-"*70)

mamiferos = kb.consultar_individuos("Mamifero")
print(f"   Mamíferos: {mamiferos}")

# ═══════════════════════════════════════════════════════════════════
# 4. VISUALIZAÇÃO DA HIERARQUIA
# ═══════════════════════════════════════════════════════════════════

print("\n📊 GERANDO VISUALIZAÇÃO...")

import matplotlib.pyplot as plt
import networkx as nx

fig, ax = plt.subplots(figsize=(14, 10))

# Criar grafo da hierarquia
G_hierarquia = nx.DiGraph()

# Adicionar conceitos e relações de superclasse
for conceito_nome, conceito in kb.conceitos.items():
    G_hierarquia.add_node(conceito_nome)
    for superclasse in conceito.superclasses:
        G_hierarquia.add_edge(superclasse, conceito_nome)

# Layout hierárquico
pos = nx.spring_layout(G_hierarquia, k=2, iterations=50, seed=42)

# Cores por nível
cores = []
for no in G_hierarquia.nodes():
    if no == "SerVivo":
        cores.append('gold')
    elif no in ["Animal", "Planta"]:
        cores.append('lightcoral')
    elif no in ["Vertebrado", "Invertebrado"]:
        cores.append('lightblue')
    elif no in ["Mamifero", "Ave", "Reptil"]:
        cores.append('lightgreen')
    elif no in ["Carnivoro", "Herbivoro", "Onivoro"]:
        cores.append('plum')
    else:
        cores.append('wheat')

nx.draw_networkx_nodes(G_hierarquia, pos, node_color=cores, 
                      node_size=2500, ax=ax)
nx.draw_networkx_labels(G_hierarquia, pos, font_size=9, 
                       font_weight='bold', ax=ax)
nx.draw_networkx_edges(G_hierarquia, pos, edge_color='gray', 
                      arrows=True, arrowsize=15, width=2, ax=ax)

ax.set_title("Hierarquia de Conceitos (TBox) - Description Logic", 
            fontsize=14, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════
# 5. RELATÓRIO FINAL
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("RELATÓRIO FINAL - DESCRIPTION LOGIC")
print("="*70)

print(f"\n🏗️ COMPONENTES:")
print(f"   • TBox (Terminologia): {len(kb.conceitos)} conceitos")
print(f"   • ABox (Assertions): {len(kb.individuos)} indivíduos")
print(f"   • Reasoning Services: Subsumption, Classification, Query")

print(f"\n🧠 REASONING SERVICES:")
print(f"   • Subsumption: C ⊑ D (C é subconceito de D)")
print(f"   • Classification: Inferir classes de indivíduos")
print(f"   • Consistency: Verificar contradições")
print(f"   • Satisfiability: Conceito tem instâncias possíveis")

print(f"\n🎯 APLICAÇÕES REAIS:")
print(f"   • Ontologias (Semantic Web)")
print(f"   • Sistemas de classificação (taxonomias)")
print(f"   • Medicina (SNOMED CT)")
print(f"   • Bioinformática (Gene Ontology)")
print(f"   • E-commerce (catálogos de produtos)")

print(f"\n💡 DL vs OUTROS:")
print(f"   • vs Lógica de 1ª Ordem: Menos expressivo, mas decidível")
print(f"   • vs Frames: DL tem semântica formal")
print(f"   • vs OWL: OWL é baseado em DL (OWL-DL)")
print(f"   • vs Redes Semânticas: DL tem reasoning rigoroso")

print(f"\n🔬 EXPRESSIVIDADE (Linguagens DL):")
print(f"   • ALC: Attributive Language with Complements")
print(f"   • SHOIN: Base do OWL-DL")
print(f"   • SROIQ: Base do OWL 2")
print(f"   • Tradeoff: Expressividade vs Complexidade computacional")

print(f"\n✅ DESCRIÇÃO LÓGICA COMPLETA!")
