# GO0311-BfsVsAAnáliseEmpírica
from collections import deque
import heapq

class ProblemaLabirinto:
    """Modelagem do problema de busca em labirinto"""
    def __init__(self, labirinto, inicio, objetivo):
        self.labirinto = labirinto
        self.linhas = len(labirinto)
        self.colunas = len(labirinto[0])
        self.estado_inicial = inicio
        self.objetivo = objetivo

    def teste_objetivo(self, estado):
        return estado == self.objetivo

    def acoes(self, estado):
        linha, coluna = estado
        acoes_possiveis = []
        movimentos = [('Norte', (-1, 0)), ('Sul', (1, 0)), 
                      ('Leste', (0, 1)), ('Oeste', (0, -1))]
        for nome, (dl, dc) in movimentos:
            nova_linha, nova_coluna = linha + dl, coluna + dc
            if (0 <= nova_linha < self.linhas and 
                0 <= nova_coluna < self.colunas and
                self.labirinto[nova_linha][nova_coluna] == 0):
                acoes_possiveis.append(nome)
        return acoes_possiveis

    def resultado(self, estado, acao):
        linha, coluna = estado
        if acao == 'Norte': return (linha - 1, coluna)
        elif acao == 'Sul': return (linha + 1, coluna)
        elif acao == 'Leste': return (linha, coluna + 1)
        elif acao == 'Oeste': return (linha, coluna - 1)

    def custo(self, estado, acao):
        return 1

    def heuristica_manhattan(self, estado):
        return abs(estado[0] - self.objetivo[0]) + abs(estado[1] - self.objetivo[1])

class No:
    """Nó da árvore de busca"""
    def __init__(self, estado, pai=None, acao=None, custo=0):
        self.estado = estado
        self.pai = pai
        self.acao = acao
        self.custo = custo

    def caminho(self):
        no_atual = self
        caminho = []
        while no_atual:
            caminho.append(no_atual.estado)
            no_atual = no_atual.pai
        return list(reversed(caminho))

def busca_largura_com_stats(problema):
    """BFS com coleta de estatísticas"""
    no_inicial = No(problema.estado_inicial)
    if problema.teste_objetivo(no_inicial.estado):
        return [no_inicial.estado], {'nos_explorados': 0, 'custo_total': 0}

    fronteira = deque([no_inicial])
    explorados = set()
    nos_explorados = 0
    max_fronteira = 1

    while fronteira:
        max_fronteira = max(max_fronteira, len(fronteira))
        no = fronteira.popleft()

        if no.estado in explorados:
            continue

        explorados.add(no.estado)
        nos_explorados += 1

        for acao in problema.acoes(no.estado):
            filho_estado = problema.resultado(no.estado, acao)
            if filho_estado not in explorados:
                custo = no.custo + problema.custo(no.estado, acao)
                filho = No(filho_estado, no, acao, custo)

                if problema.teste_objetivo(filho_estado):
                    caminho = filho.caminho()
                    return caminho, {
                        'nos_explorados': nos_explorados,
                        'nos_fronteira_max': max_fronteira,
                        'custo_total': filho.custo,
                        'comprimento_caminho': len(caminho)
                    }
                fronteira.append(filho)

    return None, {'nos_explorados': nos_explorados, 'custo_total': 0}

def busca_a_estrela(problema):
    """A* com coleta de estatísticas"""
    no_inicial = No(problema.estado_inicial)
    fronteira = []
    contador = 0
    h = problema.heuristica_manhattan(no_inicial.estado)
    heapq.heappush(fronteira, (h, contador, no_inicial))
    contador += 1

    explorados = set()
    custos = {no_inicial.estado: 0}
    max_fronteira = 1
    nos_explorados = 0

    while fronteira:
        max_fronteira = max(max_fronteira, len(fronteira))
        _, _, no = heapq.heappop(fronteira)

        if problema.teste_objetivo(no.estado):
            caminho = no.caminho()
            return caminho, {
                'nos_explorados': nos_explorados,
                'nos_fronteira_max': max_fronteira,
                'custo_total': no.custo,
                'comprimento_caminho': len(caminho)
            }

        if no.estado in explorados:
            continue

        explorados.add(no.estado)
        nos_explorados += 1

        for acao in problema.acoes(no.estado):
            filho_estado = problema.resultado(no.estado, acao)
            g = no.custo + problema.custo(no.estado, acao)

            if filho_estado in explorados:
                continue

            if filho_estado not in custos or g < custos[filho_estado]:
                custos[filho_estado] = g
                h = problema.heuristica_manhattan(filho_estado)
                f = g + h
                filho = No(filho_estado, no, acao, g)
                heapq.heappush(fronteira, (f, contador, filho))
                contador += 1

    return None, {'nos_explorados': nos_explorados, 'custo_total': 0}

# Criar labirinto 7x7 para comparação


if __name__ == "__main__":
    labirinto_comparacao = [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0]
    ]
    problema = ProblemaLabirinto(labirinto_comparacao, inicio=(0, 0), objetivo=(6, 6))

    # Executar ambos
    print("="*60)
    print("COMPARAÇÃO: BFS vs A*")
    print("="*60)

    # BFS
    print("\n🔵 Executando BFS...")
    caminho_bfs, stats_bfs = busca_largura_com_stats(problema)

    # A*
    print("\n⭐ Executando A*...")
    caminho_astar, stats_astar = busca_a_estrela(problema)

    # Tabela comparativa
    print("\n┌─────────────────────┬─────────────┬─────────────┐")
    print("│   Métrica           │     BFS     │     A*      │")
    print("├─────────────────────┼─────────────┼─────────────┤")
    print(f"│ Nós explorados      │{stats_bfs['nos_explorados']:^13}│{stats_astar['nos_explorados']:^13}│")
    print(f"│ Comprimento caminho │{len(caminho_bfs):^13}│{len(caminho_astar):^13}│")
    print(f"│ Custo total         │{stats_bfs['custo_total']:^13}│{stats_astar['custo_total']:^13}│")
    print("└─────────────────────┴─────────────┴─────────────┘")

    # Análise
    reducao = (1 - stats_astar['nos_explorados'] / stats_bfs['nos_explorados']) * 100
    print(f"\n✅ A* explorou {reducao:.1f}% MENOS nós que BFS!")

    # PERGUNTA PARA REFLEXÃO:
    # Por que A* é mais eficiente mesmo encontrando o mesmo caminho ótimo?
