# GO0313-PuzzleOitoComAEstrela
import heapq
from typing import List, Tuple, Optional
import numpy as np

print("="*70)
print("8-PUZZLE SOLVER COM A*")
print("="*70)

class EstadoPuzzle:
    """Representa um estado do 8-puzzle"""

    def __init__(self, tabuleiro: List[List[int]], movimentos: int = 0, anterior=None):
        self.tabuleiro = [linha[:] for linha in tabuleiro]
        self.movimentos = movimentos  # g(n)
        self.anterior = anterior
        self.pos_vazio = self._encontrar_vazio()

    def _encontrar_vazio(self) -> Tuple[int, int]:
        """Encontra posição do espaço vazio (0)"""
        for i in range(3):
            for j in range(3):
                if self.tabuleiro[i][j] == 0:
                    return (i, j)
        return (0, 0)

    def heuristica_manhattan(self, objetivo: 'EstadoPuzzle') -> int:
        """
        Distância Manhattan: soma das distâncias de cada peça
        até sua posição correta
        """
        distancia = 0
        for i in range(3):
            for j in range(3):
                valor = self.tabuleiro[i][j]
                if valor != 0:  # Ignorar espaço vazio
                    # Encontrar posição correta no objetivo
                    for oi in range(3):
                        for oj in range(3):
                            if objetivo.tabuleiro[oi][oj] == valor:
                                distancia += abs(i - oi) + abs(j - oj)
        return distancia

    def movimentos_possiveis(self) -> List[Tuple[int, int]]:
        """Retorna lista de direções possíveis para mover o vazio"""
        i, j = self.pos_vazio
        movimentos = []

        if i > 0: movimentos.append((-1, 0))  # Cima
        if i < 2: movimentos.append((1, 0))   # Baixo
        if j > 0: movimentos.append((0, -1))  # Esquerda
        if j < 2: movimentos.append((0, 1))   # Direita

        return movimentos

    def aplicar_movimento(self, direcao: Tuple[int, int]) -> 'EstadoPuzzle':
        """Cria novo estado aplicando movimento"""
        di, dj = direcao
        i, j = self.pos_vazio
        novo_i, novo_j = i + di, j + dj

        # Copiar tabuleiro
        novo_tab = [linha[:] for linha in self.tabuleiro]

        # Trocar vazio com peça
        novo_tab[i][j], novo_tab[novo_i][novo_j] = \
            novo_tab[novo_i][novo_j], novo_tab[i][j]

        return EstadoPuzzle(novo_tab, self.movimentos + 1, self)

    def __eq__(self, other):
        return self.tabuleiro == other.tabuleiro

    def __hash__(self):
        return hash(str(self.tabuleiro))

    def __lt__(self, other):
        return False  # Para heapq

    def exibir(self):
        """Exibe o tabuleiro"""
        for linha in self.tabuleiro:
            print("  ", end="")
            for val in linha:
                if val == 0:
                    print("[ ]", end=" ")
                else:
                    print(f"[{val}]", end=" ")
            print()

def busca_a_estrela_puzzle(inicial: EstadoPuzzle, objetivo: EstadoPuzzle):
    """
    Resolve 8-puzzle usando A*

    f(n) = g(n) + h(n)
    g(n) = número de movimentos até agora
    h(n) = distância Manhattan até objetivo
    """

    # Fila de prioridade: (f, contador, estado)
    contador = 0
    h_inicial = inicial.heuristica_manhattan(objetivo)
    f_inicial = inicial.movimentos + h_inicial
    fila = [(f_inicial, contador, inicial)]
    contador += 1

    visitados = set()
    visitados.add(hash(inicial))

    nos_explorados = 0
    max_fila = 1

    while fila:
        max_fila = max(max_fila, len(fila))
        _, _, estado_atual = heapq.heappop(fila)
        nos_explorados += 1

        # Teste de objetivo
        if estado_atual == objetivo:
            # Reconstruir caminho
            caminho = []
            estado = estado_atual
            while estado:
                caminho.append(estado)
                estado = estado.anterior
            caminho.reverse()

            return caminho, {
                'nos_explorados': nos_explorados,
                'max_fila': max_fila,
                'movimentos': estado_atual.movimentos
            }

        # Expandir sucessores
        for direcao in estado_atual.movimentos_possiveis():
            sucessor = estado_atual.aplicar_movimento(direcao)
            hash_sucessor = hash(sucessor)

            if hash_sucessor not in visitados:
                visitados.add(hash_sucessor)
                g = sucessor.movimentos
                h = sucessor.heuristica_manhattan(objetivo)
                f = g + h
                heapq.heappush(fila, (f, contador, sucessor))
                contador += 1

    return None, {'nos_explorados': nos_explorados}

# ═══════════════════════════════════════════════════════════════════
# TESTAR COM PUZZLE FÁCIL
# ═══════════════════════════════════════════════════════════════════

print("\n🧩 PUZZLE INICIAL:")
inicial = EstadoPuzzle([
    [1, 2, 3],
    [4, 0, 5],
    [7, 8, 6]
])
inicial.exibir()

print("\n🎯 OBJETIVO:")
objetivo = EstadoPuzzle([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
])
objetivo.exibir()

print(f"\n🔍 Heurística inicial (Manhattan): {inicial.heuristica_manhattan(objetivo)}")

print("\n⏳ RESOLVENDO COM A*...")
print("-" * 70)

caminho, stats = busca_a_estrela_puzzle(inicial, objetivo)

if caminho:
    print(f"\n✅ SOLUÇÃO ENCONTRADA!")
    print(f"   Número de movimentos: {stats['movimentos']}")
    print(f"   Nós explorados: {stats['nos_explorados']}")
    print(f"   Fila máxima: {stats['max_fila']}")

    print(f"\n📋 SEQUÊNCIA DE MOVIMENTOS ({len(caminho)} estados):")
    for i, estado in enumerate(caminho):
        print(f"\n   Passo {i} (g={estado.movimentos}, h={estado.heuristica_manhattan(objetivo)}):")
        estado.exibir()
else:
    print("❌ Sem solução!")

# ═══════════════════════════════════════════════════════════════════
# PUZZLE MAIS DIFÍCIL
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("PUZZLE DIFÍCIL")
print("="*70)

print("\n🧩 PUZZLE INICIAL (DIFÍCIL):")
inicial_dificil = EstadoPuzzle([
    [7, 2, 4],
    [5, 0, 6],
    [8, 3, 1]
])
inicial_dificil.exibir()

print("\n⏳ RESOLVENDO...")
caminho_dif, stats_dif = busca_a_estrela_puzzle(inicial_dificil, objetivo)

if caminho_dif:
    print(f"\n✅ SOLUÇÃO ENCONTRADA!")
    print(f"   Movimentos: {stats_dif['movimentos']}")
    print(f"   Nós explorados: {stats_dif['nos_explorados']}")
    print(f"   Eficiência: {stats_dif['movimentos']}/{stats_dif['nos_explorados']} = {stats_dif['movimentos']/stats_dif['nos_explorados']:.2%}")
