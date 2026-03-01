# GO0216-RedesSemânticas
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict, deque
import networkx as nx
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════
# 1. REDE SEMÂNTICA - FUNDAMENTOS
# ═══════════════════════════════════════════════════════════════════

print("="*70)
print("REDES SEMÂNTICAS - SISTEMA DE RECOMENDAÇÃO DE FILMES")
print("="*70)

class RedeSemantica:
    """
    Rede Semântica: Grafo de conceitos e relações

    Tipos de relações:
    - IS-A: Hierarquia de classes (ex: cachorro IS-A animal)
    - PART-OF: Composição (ex: roda PART-OF carro)
    - HAS-PROPERTY: Atributos (ex: cachorro HAS-PROPERTY pernas=4)
    - INSTANCE-OF: Instância de classe
    - Relações customizadas (ex: ATUA-EM, DIRIGIDO-POR)
    """

    def __init__(self):
        # Grafo direcionado: (origem, relacao, destino)
        self.grafo = nx.MultiDiGraph()

        # Índices para busca rápida
        self.relacoes_por_tipo: Dict[str, List[Tuple]] = defaultdict(list)
        self.propriedades: Dict[str, Dict[str, Any]] = defaultdict(dict)

    def adicionar_no(self, conceito: str, tipo: str = "conceito"):
        """Adicionar nó (conceito) à rede"""
        self.grafo.add_node(conceito, tipo=tipo)

    def adicionar_relacao(self, origem: str, relacao: str, destino: str):
        """Adicionar relação entre conceitos"""
        # Adicionar nós se não existem
        if origem not in self.grafo:
            self.adicionar_no(origem)
        if destino not in self.grafo:
            self.adicionar_no(destino)

        # Adicionar aresta
        self.grafo.add_edge(origem, destino, relacao=relacao)

        # Indexar
        self.relacoes_por_tipo[relacao].append((origem, destino))

    def adicionar_propriedade(self, conceito: str, propriedade: str, valor: Any):
        """Adicionar propriedade a conceito"""
        if conceito not in self.grafo:
            self.adicionar_no(conceito)

        self.propriedades[conceito][propriedade] = valor

        # Também como aresta para visualização
        no_prop = f"{conceito}_{propriedade}={valor}"
        self.grafo.add_node(no_prop, tipo="propriedade")
        self.grafo.add_edge(conceito, no_prop, relacao="HAS-PROPERTY")

    def obter_propriedade(self, conceito: str, propriedade: str, 
                         herdar: bool = True) -> Optional[Any]:
        """
        Obter propriedade de conceito (com herança)

        Args:
            conceito: Nome do conceito
            propriedade: Nome da propriedade
            herdar: Se True, busca em superclasses (IS-A)
        """
        # Propriedade direta
        if conceito in self.propriedades:
            if propriedade in self.propriedades[conceito]:
                return self.propriedades[conceito][propriedade]

        # Herança via IS-A
        if herdar:
            superclasses = self.obter_relacionados(conceito, "IS-A")
            for superclasse in superclasses:
                valor = self.obter_propriedade(superclasse, propriedade, herdar=True)
                if valor is not None:
                    return valor

        return None

    def obter_relacionados(self, conceito: str, relacao: str, 
                          direcao: str = "saida") -> List[str]:
        """
        Obter conceitos relacionados

        Args:
            conceito: Conceito origem
            relacao: Tipo de relação
            direcao: 'saida' (origem→destino) ou 'entrada' (destino←origem)
        """
        relacionados = []

        if direcao == "saida":
            for vizinho in self.grafo.successors(conceito):
                dados_aresta = self.grafo.get_edge_data(conceito, vizinho)
                if dados_aresta:
                    for key, attrs in dados_aresta.items():
                        if attrs.get("relacao") == relacao:
                            relacionados.append(vizinho)
        else:  # entrada
            for vizinho in self.grafo.predecessors(conceito):
                dados_aresta = self.grafo.get_edge_data(vizinho, conceito)
                if dados_aresta:
                    for key, attrs in dados_aresta.items():
                        if attrs.get("relacao") == relacao:
                            relacionados.append(vizinho)

        return relacionados

    def spreading_activation(self, conceito_origem: str, 
                            max_profundidade: int = 3) -> Dict[str, float]:
        """
        Spreading Activation: Propagar ativação pela rede

        Usado para encontrar conceitos relacionados (similaridade)

        Args:
            conceito_origem: Conceito inicial
            max_profundidade: Máximo de saltos

        Returns:
            Dict {conceito: ativação} (0-1)
        """
        ativacoes = {conceito_origem: 1.0}
        fila = deque([(conceito_origem, 1.0, 0)])  # (conceito, ativacao, profundidade)
        visitados = {conceito_origem}

        while fila:
            conceito_atual, ativacao_atual, profundidade = fila.popleft()

            if profundidade >= max_profundidade:
                continue

            # Propagar para vizinhos
            for vizinho in self.grafo.neighbors(conceito_atual):
                if vizinho not in visitados:
                    # Decaimento: ativação reduz com distância
                    nova_ativacao = ativacao_atual * 0.7

                    ativacoes[vizinho] = ativacoes.get(vizinho, 0) + nova_ativacao
                    fila.append((vizinho, nova_ativacao, profundidade + 1))
                    visitados.add(vizinho)

        return ativacoes

    def encontrar_caminho(self, origem: str, destino: str, 
                         max_profundidade: int = 5) -> Optional[List[Tuple[str, str]]]:
        """
        Encontrar caminho entre dois conceitos

        Returns:
            Lista de (conceito, relacao) ou None
        """
        if origem not in self.grafo or destino not in self.grafo:
            return None

        try:
            # Caminho mais curto
            caminho_nos = nx.shortest_path(self.grafo.to_undirected(), 
                                           origem, destino)

            # Obter relações
            caminho_completo = []
            for i in range(len(caminho_nos) - 1):
                no_atual = caminho_nos[i]
                proximo_no = caminho_nos[i + 1]

                # Pegar relação (pode ter múltiplas)
                dados = self.grafo.get_edge_data(no_atual, proximo_no)
                if not dados:
                    # Tentar direção inversa
                    dados = self.grafo.get_edge_data(proximo_no, no_atual)

                if dados:
                    relacao = list(dados.values())[0].get("relacao", "?")
                    caminho_completo.append((no_atual, relacao))

            caminho_completo.append((caminho_nos[-1], None))

            return caminho_completo

        except nx.NetworkXNoPath:
            return None

    def resumo(self):
        """Resumo da rede"""
        return {
            "nos": self.grafo.number_of_nodes(),
            "arestas": self.grafo.number_of_edges(),
            "tipos_relacao": len(self.relacoes_por_tipo),
            "conceitos_com_propriedades": len(self.propriedades)
        }

# ═══════════════════════════════════════════════════════════════════
# 2. CONSTRUIR REDE SEMÂNTICA DE FILMES
# ═══════════════════════════════════════════════════════════════════

print("\n🎬 CONSTRUINDO REDE SEMÂNTICA DE FILMES...")

rede = RedeSemantica()

# Hierarquia de gêneros (IS-A)
rede.adicionar_relacao("Filme", "IS-A", "Midia")
rede.adicionar_relacao("Acao", "IS-A", "Filme")
rede.adicionar_relacao("Drama", "IS-A", "Filme")
rede.adicionar_relacao("Ficcao_Cientifica", "IS-A", "Filme")
rede.adicionar_relacao("Comedia", "IS-A", "Filme")

print("   ✅ Hierarquia de gêneros (IS-A)")

# Filmes específicos (INSTANCE-OF)
filmes = {
    "Matrix": "Ficcao_Cientifica",
    "Inception": "Ficcao_Cientifica",
    "Forrest_Gump": "Drama",
    "Die_Hard": "Acao",
    "The_Hangover": "Comedia"
}

for filme, genero in filmes.items():
    rede.adicionar_relacao(filme, "INSTANCE-OF", genero)

print(f"   ✅ {len(filmes)} filmes adicionados (INSTANCE-OF)")

# Atores e relações (ATUA-EM)
atores_filmes = [
    ("Keanu_Reeves", "Matrix"),
    ("Keanu_Reeves", "John_Wick"),
    ("Leonardo_DiCaprio", "Inception"),
    ("Tom_Hanks", "Forrest_Gump"),
    ("Bruce_Willis", "Die_Hard"),
    ("Bradley_Cooper", "The_Hangover"),
]

for ator, filme in atores_filmes:
    rede.adicionar_relacao(ator, "ATUA-EM", filme)

print(f"   ✅ {len(atores_filmes)} relações ator-filme (ATUA-EM)")

# Diretores (DIRIGIDO-POR)
diretores = [
    ("Matrix", "Wachowskis"),
    ("Inception", "Christopher_Nolan"),
    ("Forrest_Gump", "Robert_Zemeckis"),
    ("Die_Hard", "John_McTiernan"),
]

for filme, diretor in diretores:
    rede.adicionar_relacao(filme, "DIRIGIDO-POR", diretor)

print(f"   ✅ {len(diretores)} relações filme-diretor (DIRIGIDO-POR)")

# Propriedades (HAS-PROPERTY)
propriedades_filmes = {
    "Matrix": {"ano": 1999, "duracao": 136, "rating": 8.7},
    "Inception": {"ano": 2010, "duracao": 148, "rating": 8.8},
    "Forrest_Gump": {"ano": 1994, "duracao": 142, "rating": 8.8},
    "Die_Hard": {"ano": 1988, "duracao": 132, "rating": 8.2},
}

for filme, props in propriedades_filmes.items():
    for prop, valor in props.items():
        rede.adicionar_propriedade(filme, prop, valor)

print(f"   ✅ Propriedades adicionadas (HAS-PROPERTY)")

# Temas (TEM-TEMA)
temas = [
    ("Matrix", "Realidade_Virtual"),
    ("Matrix", "IA"),
    ("Inception", "Sonhos"),
    ("Inception", "Subconsciente"),
    ("Ficcao_Cientifica", "Tecnologia"),
]

for filme, tema in temas:
    rede.adicionar_relacao(filme, "TEM-TEMA", tema)

print(f"   ✅ {len(temas)} relações de temas (TEM-TEMA)")

resumo_rede = rede.resumo()
print(f"\n📊 RESUMO DA REDE:")
print(f"   Nós: {resumo_rede['nos']}")
print(f"   Arestas: {resumo_rede['arestas']}")
print(f"   Tipos de relação: {resumo_rede['tipos_relacao']}")

# ═══════════════════════════════════════════════════════════════════
# 3. CONSULTAS E INFERÊNCIAS
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("CONSULTAS E INFERÊNCIAS")
print("="*70)

# 1. Herança de propriedades
print("\n🔍 CONSULTA 1: Herança de Propriedades")
print("-"*70)

filme_teste = "Matrix"
prop_teste = "rating"

valor = rede.obter_propriedade(filme_teste, prop_teste, herdar=True)
print(f"   {filme_teste}.{prop_teste} = {valor}")

# Adicionar propriedade no gênero
rede.adicionar_propriedade("Ficcao_Cientifica", "publico_alvo", "adultos")

valor_herdado = rede.obter_propriedade("Matrix", "publico_alvo", herdar=True)
print(f"   {filme_teste}.publico_alvo = {valor_herdado} (herdado de Ficcao_Cientifica)")

# 2. Spreading Activation (filmes similares)
print("\n🔍 CONSULTA 2: Spreading Activation (Recomendação)")
print("-"*70)

filme_origem = "Matrix"
ativacoes = rede.spreading_activation(filme_origem, max_profundidade=3)

# Filtrar apenas filmes (INSTANCE-OF algum gênero)
filmes_ativados = {conceito: ativ for conceito, ativ in ativacoes.items()
                   if conceito in filmes.keys() and conceito != filme_origem}

# Top 3 mais ativados
top_recomendacoes = sorted(filmes_ativados.items(), key=lambda x: x[1], reverse=True)[:3]

print(f"   Se você gostou de '{filme_origem}', recomendamos:")
for i, (filme_rec, ativ) in enumerate(top_recomendacoes, 1):
    print(f"      {i}. {filme_rec} (ativação: {ativ:.2f})")

# 3. Encontrar caminho entre conceitos
print("\n🔍 CONSULTA 3: Caminho Entre Conceitos")
print("-"*70)

origem_caminho = "Keanu_Reeves"
destino_caminho = "Christopher_Nolan"

caminho = rede.encontrar_caminho(origem_caminho, destino_caminho)

if caminho:
    print(f"   Caminho de '{origem_caminho}' até '{destino_caminho}':")
    for i, (conceito, relacao) in enumerate(caminho):
        if relacao:
            print(f"      {conceito} --[{relacao}]--> ", end="")
        else:
            print(f"{conceito}")
else:
    print(f"   ❌ Nenhum caminho encontrado")

# 4. Quais filmes têm tema "IA"?
print("\n🔍 CONSULTA 4: Filmes com Tema 'IA'")
print("-"*70)

filmes_ia = rede.obter_relacionados("IA", "TEM-TEMA", direcao="entrada")
print(f"   Filmes com tema 'IA': {filmes_ia}")

# 5. Quais atores atuaram em filmes de Ficção Científica?
print("\n🔍 CONSULTA 5: Atores de Ficção Científica")
print("-"*70)

# Pegar filmes de ficção científica
filmes_ficcao = rede.obter_relacionados("Ficcao_Cientifica", "INSTANCE-OF", direcao="entrada")

atores_ficcao = set()
for filme in filmes_ficcao:
    atores_filme = rede.obter_relacionados(filme, "ATUA-EM", direcao="entrada")
    atores_ficcao.update(atores_filme)

print(f"   Atores que atuaram em Ficção Científica: {list(atores_ficcao)}")

# ═══════════════════════════════════════════════════════════════════
# 4. VISUALIZAÇÃO
# ═══════════════════════════════════════════════════════════════════

print("\n📊 GERANDO VISUALIZAÇÃO...")

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Subgrafo 1: Hierarquia de gêneros
ax1 = axes[0]
subgrafo_generos = nx.DiGraph()

for origem, destino in rede.relacoes_por_tipo["IS-A"]:
    subgrafo_generos.add_edge(origem, destino)

pos1 = nx.spring_layout(subgrafo_generos, k=2, iterations=50)
nx.draw_networkx_nodes(subgrafo_generos, pos1, node_color='lightblue', 
                       node_size=2000, ax=ax1)
nx.draw_networkx_labels(subgrafo_generos, pos1, font_size=10, ax=ax1)
nx.draw_networkx_edges(subgrafo_generos, pos1, edge_color='gray', 
                       arrows=True, arrowsize=20, ax=ax1)
ax1.set_title("Hierarquia de Gêneros (IS-A)", fontsize=14, fontweight='bold')
ax1.axis('off')

# Subgrafo 2: Rede completa (simplificada)
ax2 = axes[1]

# Limitar a visualização (muitos nós poluem)
nos_principais = list(filmes.keys()) + ["Keanu_Reeves", "Leonardo_DiCaprio", 
                                         "Ficcao_Cientifica", "Drama", "Acao"]
subgrafo_principal = rede.grafo.subgraph(nos_principais)

pos2 = nx.spring_layout(subgrafo_principal, k=3, iterations=50)

# Cores por tipo
cores = []
for no in subgrafo_principal.nodes():
    if no in filmes.keys():
        cores.append('gold')
    elif "_" in no and no.split("_")[0] in ["Keanu", "Leonardo"]:
        cores.append('lightcoral')
    else:
        cores.append('lightgreen')

nx.draw_networkx_nodes(subgrafo_principal, pos2, node_color=cores, 
                       node_size=1500, ax=ax2)
nx.draw_networkx_labels(subgrafo_principal, pos2, font_size=8, ax=ax2)

# Arestas com rótulos
arestas_rotulos = {}
for u, v, data in subgrafo_principal.edges(data=True):
    relacao = data.get('relacao', '')
    arestas_rotulos[(u, v)] = relacao

nx.draw_networkx_edges(subgrafo_principal, pos2, edge_color='gray', 
                       arrows=True, arrowsize=15, ax=ax2)
nx.draw_networkx_edge_labels(subgrafo_principal, pos2, arestas_rotulos, 
                             font_size=7, ax=ax2)

ax2.set_title("Rede Semântica de Filmes (Simplificada)", fontsize=14, fontweight='bold')
ax2.axis('off')

plt.suptitle("Redes Semânticas - Representação de Conhecimento", 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════
# 5. RELATÓRIO FINAL
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("RELATÓRIO FINAL - REDES SEMÂNTICAS")
print("="*70)

print(f"\n🏗️ TIPOS DE RELAÇÕES:")
for tipo_rel, lista_rel in rede.relacoes_por_tipo.items():
    print(f"   • {tipo_rel}: {len(lista_rel)} relações")

print(f"\n🧠 CAPACIDADES:")
print(f"   • Herança: Propriedades propagam por IS-A")
print(f"   • Spreading Activation: Encontra conceitos similares")
print(f"   • Path Finding: Caminho entre conceitos")
print(f"   • Inferência: Deduz relações transitivas")

print(f"\n🎯 APLICAÇÕES REAIS:")
print(f"   • Sistemas de recomendação (filmes, produtos)")
print(f"   • Processamento de linguagem natural")
print(f"   • Sistemas de perguntas e respostas (Q&A)")
print(f"   • Mecanismos de busca (Google Knowledge Graph)")
print(f"   • Assistentes virtuais (Siri, Alexa)")

print(f"\n💡 REDES SEMÂNTICAS vs OUTROS:")
print(f"   • vs Ontologias: Menos formais, mais flexíveis")
print(f"   • vs Knowledge Graphs: Similar (KG é versão moderna)")
print(f"   • vs Frames: Redes são grafos, Frames são estruturas")
print(f"   • vs Lógica: Menos expressivos, mas mais intuitivos")

print(f"\n🔬 CONCEITOS-CHAVE:")
print(f"   • Nós: Conceitos (entidades, classes)")
print(f"   • Arestas: Relações (IS-A, PART-OF, HAS-PROPERTY)")
print(f"   • Herança: Propagação via IS-A")
print(f"   • Spreading Activation: Similaridade via propagação")

print("\n✅ REDE SEMÂNTICA COMPLETA!")
