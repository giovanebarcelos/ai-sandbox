# GO0309-ImplementarOAlgoritmoCompleto
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
    def __init__(self, estado, pai=None, acao=None, custo_caminho=0):
        self.estado = estado
        self.pai = pai
        self.acao = acao
        self.custo_caminho = custo_caminho
        self.profundidade = 0 if pai is None else pai.profundidade + 1

    def caminho(self):
        """Reconstrói caminho até a raiz"""
        no_atual = self
        caminho = []
        while no_atual:
            caminho.append(no_atual.estado)
            no_atual = no_atual.pai
        return list(reversed(caminho))

def busca_a_estrela(problema):
    """
    Implementação completa do A*

    Returns:
        (caminho, estatísticas) onde:
        caminho = lista de estados
        estatísticas = dict com métricas
    """
    # 1. Inicializar
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

    # 2. Loop principal
    while fronteira:
        max_fronteira = max(max_fronteira, len(fronteira))
        _, _, no = heapq.heappop(fronteira)

        # Testar objetivo
        if problema.teste_objetivo(no.estado):
            caminho = no.caminho()
            estatisticas = {
                'nos_explorados': nos_explorados,
                'nos_fronteira_max': max_fronteira,
                'custo_total': no.custo_caminho,
                'comprimento_caminho': len(caminho)
            }
            return caminho, estatisticas

        if no.estado in explorados:
            continue

        explorados.add(no.estado)
        nos_explorados += 1

        # Expandir sucessores
        for acao in problema.acoes(no.estado):
            filho_estado = problema.resultado(no.estado, acao)
            g = no.custo_caminho + problema.custo(no.estado, acao)

            if filho_estado in explorados:
                continue

            if filho_estado not in custos or g < custos[filho_estado]:
                custos[filho_estado] = g
                h = problema.heuristica_manhattan(filho_estado)
                f = g + h
                filho = No(filho_estado, no, acao, g)
                heapq.heappush(fronteira, (f, contador, filho))
                contador += 1

    # Sem solução
    estatisticas = {
        'nos_explorados': nos_explorados,
        'nos_fronteira_max': max_fronteira,
        'custo_total': 0,
        'comprimento_caminho': 0
    }
    return None, estatisticas

# Criar labirinto de exemplo para testar
labirinto_teste = [
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 1, 1, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 1, 0, 0, 0, 1, 0, 0, 0]
]

# Criar problema: início (0,0) até objetivo (9,9)
problema = ProblemaLabirinto(labirinto_teste, inicio=(0, 0), objetivo=(9, 9))

# Testar
print("🚀 Executando A* no labirinto 10x10...")
caminho, stats = busca_a_estrela(problema)
if caminho:
    print(f"✅ Caminho encontrado: {len(caminho)} estados")
    print(f"   Nós explorados: {stats['nos_explorados']}")
    print(f"   Custo total: {stats['custo_total']}")
    print(f"   Fronteira máxima: {stats['nos_fronteira_max']}")
else:
    print("❌ Sem solução encontrada")
