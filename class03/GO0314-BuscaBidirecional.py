# GO0314-BuscaBidirecional
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

print("="*70)
print("BUSCA BIDIRECIONAL - OTIMIZAÇÃO DE BFS")
print("="*70)

class GrafoBidirecional:
    """Grafo para demonstrar busca bidirecional"""

    def __init__(self):
        self.grafo = {}

    def adicionar_aresta(self, origem: str, destino: str):
        """Adiciona aresta bidirecional"""
        if origem not in self.grafo:
            self.grafo[origem] = []
        if destino not in self.grafo:
            self.grafo[destino] = []

        if destino not in self.grafo[origem]:
            self.grafo[origem].append(destino)
        if origem not in self.grafo[destino]:
            self.grafo[destino].append(origem)

    def vizinhos(self, no: str) -> List[str]:
        """Retorna vizinhos de um nó"""
        return self.grafo.get(no, [])

    def bfs_unidirecional(self, inicio: str, objetivo: str) -> Tuple[Optional[List[str]], Dict]:
        """BFS tradicional (unidirecional) para comparação"""
        if inicio == objetivo:
            return [inicio], {'nos_explorados': 0}

        fila = deque([(inicio, [inicio])])
        visitados = {inicio}
        nos_explorados = 0

        while fila:
            no_atual, caminho = fila.popleft()
            nos_explorados += 1

            for vizinho in self.vizinhos(no_atual):
                if vizinho not in visitados:
                    novo_caminho = caminho + [vizinho]

                    if vizinho == objetivo:
                        return novo_caminho, {'nos_explorados': nos_explorados}

                    visitados.add(vizinho)
                    fila.append((vizinho, novo_caminho))

        return None, {'nos_explorados': nos_explorados}

    def bfs_bidirecional(self, inicio: str, objetivo: str) -> Tuple[Optional[List[str]], Dict]:
        """
        Busca bidirecional: executa BFS do início e do fim simultaneamente
        até as fronteiras se encontrarem
        """
        if inicio == objetivo:
            return [inicio], {'nos_explorados_inicio': 0, 'nos_explorados_fim': 0}

        # Fronteiras
        fila_inicio = deque([(inicio, [inicio])])
        fila_fim = deque([(objetivo, [objetivo])])

        # Visitados com caminhos
        visitados_inicio = {inicio: [inicio]}
        visitados_fim = {objetivo: [objetivo]}

        nos_explorados_inicio = 0
        nos_explorados_fim = 0

        while fila_inicio and fila_fim:
            # Expandir da fronteira do início
            if fila_inicio:
                no_atual, caminho = fila_inicio.popleft()
                nos_explorados_inicio += 1

                for vizinho in self.vizinhos(no_atual):
                    if vizinho not in visitados_inicio:
                        novo_caminho = caminho + [vizinho]
                        visitados_inicio[vizinho] = novo_caminho

                        # INTERSEÇÃO ENCONTRADA!
                        if vizinho in visitados_fim:
                            # Unir caminhos
                            caminho_completo = novo_caminho[:-1] + list(reversed(visitados_fim[vizinho]))
                            return caminho_completo, {
                                'nos_explorados_inicio': nos_explorados_inicio,
                                'nos_explorados_fim': nos_explorados_fim,
                                'nos_explorados_total': nos_explorados_inicio + nos_explorados_fim
                            }

                        fila_inicio.append((vizinho, novo_caminho))

            # Expandir da fronteira do fim
            if fila_fim:
                no_atual, caminho = fila_fim.popleft()
                nos_explorados_fim += 1

                for vizinho in self.vizinhos(no_atual):
                    if vizinho not in visitados_fim:
                        novo_caminho = caminho + [vizinho]
                        visitados_fim[vizinho] = novo_caminho

                        # INTERSEÇÃO ENCONTRADA!
                        if vizinho in visitados_inicio:
                            # Unir caminhos
                            caminho_completo = visitados_inicio[vizinho][:-1] + list(reversed(novo_caminho))
                            return caminho_completo, {
                                'nos_explorados_inicio': nos_explorados_inicio,
                                'nos_explorados_fim': nos_explorados_fim,
                                'nos_explorados_total': nos_explorados_inicio + nos_explorados_fim
                            }

                        fila_fim.append((vizinho, novo_caminho))

        return None, {
            'nos_explorados_inicio': nos_explorados_inicio,
            'nos_explorados_fim': nos_explorados_fim,
            'nos_explorados_total': nos_explorados_inicio + nos_explorados_fim
        }

# ═══════════════════════════════════════════════════════════════════
# CRIAR GRAFO GRANDE PARA DEMONSTRAR EFICIÊNCIA
# ═══════════════════════════════════════════════════════════════════

print("\n🌐 CRIANDO GRAFO GRANDE (rede social simplificada)...")

grafo = GrafoBidirecional()

# Estrutura: camadas de conexões
# Camada 0: A
# Camada 1: B, C, D
# Camada 2: E, F, G, H, I
# Camada 3: J, K, L, M, N, O, P
# ...

# Camada 0 -> 1
grafo.adicionar_aresta("A", "B")
grafo.adicionar_aresta("A", "C")
grafo.adicionar_aresta("A", "D")

# Camada 1 -> 2
for origem in ["B", "C", "D"]:
    for i in range(3):
        destino = chr(ord("E") + (ord(origem) - ord("B"))*3 + i)
        grafo.adicionar_aresta(origem, destino)

# Camada 2 -> 3
for origem in ["E", "F", "G", "H", "I", "J"]:
    for i in range(2):
        idx = (ord(origem) - ord("E"))*2 + i
        destino = chr(ord("K") + idx)
        if ord(destino) <= ord("Z"):
            grafo.adicionar_aresta(origem, destino)

# Conexões laterais (mesma camada)
grafo.adicionar_aresta("B", "C")
grafo.adicionar_aresta("C", "D")
grafo.adicionar_aresta("E", "F")
grafo.adicionar_aresta("G", "H")
grafo.adicionar_aresta("K", "L")
grafo.adicionar_aresta("M", "N")

print(f"✅ Grafo criado: {len(grafo.grafo)} nós")

# ═══════════════════════════════════════════════════════════════════
# COMPARAR BFS UNIDIRECIONAL VS BIDIRECIONAL
# ═══════════════════════════════════════════════════════════════════

inicio = "A"
objetivo = "P"

print(f"\n🔍 BUSCA: {inicio} → {objetivo}")
print("=" * 70)

# BFS Unidirecional
print("\n📤 BFS UNIDIRECIONAL (tradicional):")
caminho_uni, stats_uni = grafo.bfs_unidirecional(inicio, objetivo)

if caminho_uni:
    print(f"   ✅ Caminho: {' → '.join(caminho_uni)}")
    print(f"   Comprimento: {len(caminho_uni)} nós")
    print(f"   Nós explorados: {stats_uni['nos_explorados']}")

# BFS Bidirecional
print("\n🔄 BFS BIDIRECIONAL (otimizado):")
caminho_bi, stats_bi = grafo.bfs_bidirecional(inicio, objetivo)

if caminho_bi:
    print(f"   ✅ Caminho: {' → '.join(caminho_bi)}")
    print(f"   Comprimento: {len(caminho_bi)} nós")
    print(f"   Nós explorados do início: {stats_bi['nos_explorados_inicio']}")
    print(f"   Nós explorados do fim: {stats_bi['nos_explorados_fim']}")
    print(f"   Total explorados: {stats_bi['nos_explorados_total']}")

# Comparação
print("\n📊 COMPARAÇÃO:")
print("-" * 70)
if caminho_uni and caminho_bi:
    reducao = ((stats_uni['nos_explorados'] - stats_bi['nos_explorados_total']) / 
               stats_uni['nos_explorados']) * 100
    print(f"   Unidirecional: {stats_uni['nos_explorados']} nós")
    print(f"   Bidirecional:  {stats_bi['nos_explorados_total']} nós")
    print(f"   🚀 Redução: {reducao:.1f}%")
    print(f"\n   Speedup: {stats_uni['nos_explorados'] / stats_bi['nos_explorados_total']:.2f}x")

# ═══════════════════════════════════════════════════════════════════
# ANÁLISE TEÓRICA
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("ANÁLISE TEÓRICA")
print("="*70)

print("""
📘 COMPLEXIDADE:

BFS Unidirecional:
   • Explora O(b^d) nós
   • b = fator de ramificação (vizinhos por nó)
   • d = profundidade da solução

BFS Bidirecional:
   • Explora O(b^(d/2)) do início + O(b^(d/2)) do fim
   • Total: O(2 * b^(d/2)) << O(b^d)

EXEMPLO (b=3, d=6):
   • Unidirecional: 3^6 = 729 nós
   • Bidirecional: 2 * 3^3 = 54 nós
   • Redução: 93% menos nós!

✅ VANTAGENS:
   • Exponencialmente mais eficiente
   • Encontra mesma solução ótima (se BFS)
   • Essencial para grafos grandes

⚠️ DESVANTAGENS:
   • Precisa conhecer o objetivo explicitamente
   • Mais complexo de implementar
   • Requer memória para ambas fronteiras
""")

# ═══════════════════════════════════════════════════════════════════
# TESTE ADICIONAL: CAMINHOS DIFERENTES
# ═══════════════════════════════════════════════════════════════════

print("="*70)
print("TESTES ADICIONAIS")
print("="*70)

testes = [
    ("A", "K"),
    ("B", "N"),
    ("C", "M")
]

for orig, dest in testes:
    _, stats_u = grafo.bfs_unidirecional(orig, dest)
    _, stats_b = grafo.bfs_bidirecional(orig, dest)

    print(f"\n{orig} → {dest}:")
    print(f"   Unidirecional: {stats_u['nos_explorados']} nós")
    print(f"   Bidirecional:  {stats_b['nos_explorados_total']} nós")
    if stats_u['nos_explorados'] > 0:
        reducao = ((stats_u['nos_explorados'] - stats_b['nos_explorados_total']) / 
                   stats_u['nos_explorados']) * 100
        print(f"   Redução: {reducao:.1f}%")
