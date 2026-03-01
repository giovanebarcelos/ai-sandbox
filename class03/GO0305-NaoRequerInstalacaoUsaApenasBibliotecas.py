# GO0305-NãoRequerInstalaçãoUsaApenasBibliotecas
import heapq

class FilaPrioridade:
    """Fila de prioridade para A*"""
    def __init__(self):
        self.heap = []
        self.contador = 0  # Para desempate

    def adicionar(self, no, prioridade):
        # heapq é min-heap (menor prioridade sai primeiro)
        entrada = (prioridade, self.contador, no)
        heapq.heappush(self.heap, entrada)
        self.contador += 1

    def remover(self):
        if self.vazia():
            raise Exception("Fila vazia")
        prioridade, _, no = heapq.heappop(self.heap)
        return no

    def vazia(self):
        return len(self.heap) == 0

def busca_a_estrela(problema, heuristica):
    """
    Algoritmo A*

    Args:
        problema: Instância do problema
        heuristica: Função h(estado) que retorna estimativa de custo

    Retorna:
        Lista de ações ou None
    """
    # Nó inicial
    no = No(problema.estado_inicial)

    # Teste trivial
    if problema.teste_objetivo(no.estado):
        return no.solucao()

    # Fronteira: fila de prioridade por f(n) = g(n) + h(n)
    fronteira = FilaPrioridade()
    f = no.custo_caminho + heuristica(no.estado)
    fronteira.adicionar(no, f)

    # Dicionário de custos: melhor g(n) conhecido para cada estado
    custos = {no.estado: no.custo_caminho}

    # Estados explorados
    explorados = set()
    nos_explorados = 0

    while not fronteira.vazia():
        # Remove nó com menor f(n)
        no = fronteira.remover()
        nos_explorados += 1

        # Testa objetivo
        if problema.teste_objetivo(no.estado):
            print(f"✅ A* encontrou solução ótima!")
            print(f"   Nós explorados: {nos_explorados}")
            print(f"   Custo do caminho: {no.custo_caminho}")
            print(f"   Comprimento: {len(no.solucao())} ações")
            return no.solucao()

        # Marca como explorado
        explorados.add(no.estado)

        # Expande nó
        for acao in problema.acoes(no.estado):
            filho = No(
                estado=problema.resultado(no.estado, acao),
                pai=no,
                acao=acao,
                custo_caminho=no.custo_caminho + problema.custo(no.estado, acao)
            )

            # Verifica se já foi explorado
            if filho.estado in explorados:
                continue

            # Calcula f(n) = g(n) + h(n)
            g = filho.custo_caminho
            h = heuristica(filho.estado)
            f = g + h

            # Adiciona se for novo OU se encontrou caminho melhor
            if filho.estado not in custos or g < custos[filho.estado]:
                custos[filho.estado] = g
                fronteira.adicionar(filho, f)

    print(f"❌ Sem solução (explorou {nos_explorados} nós)")
    return None

# Exemplo: A* em grafo com custos e heurística Manhattan
class ProblemaGrafoComCusto:
    def __init__(self, inicial, objetivo, grafo, posicoes):
        self.estado_inicial = inicial
        self.objetivo = objetivo
        self.grafo = grafo  # {estado: [(vizinho, custo), ...]}
        self.posicoes = posicoes  # {estado: (x, y)}

    def teste_objetivo(self, estado):
        return estado == self.objetivo

    def acoes(self, estado):
        return [v for v, _ in self.grafo.get(estado, [])]

    def resultado(self, estado, acao):
        return acao

    def custo(self, estado, acao):
        for vizinho, custo in self.grafo.get(estado, []):
            if vizinho == acao:
                return custo
        return 1

# Grafo: S→A(5)→B(1)→G(2), S→C(4)→D(3)


if __name__ == "__main__":
    grafo = {
        'S': [('A', 5), ('C', 4)],
        'A': [('B', 1), ('D', 2)],
        'B': [('G', 2), ('E', 1)],
        'C': [('D', 3)],
        'D': [('E', 4)],
        'E': [], 'G': []
    }
    # Posições para heurística Manhattan
    posicoes = {'S': (0, 0), 'A': (1, 0), 'B': (2, 0), 'G': (3, 0),
                'C': (1, 2), 'D': (2, 2), 'E': (3, 2)}

    # Heurística: distância Manhattan até G
    def heuristica_manhattan(estado):
        x1, y1 = posicoes[estado]
        x2, y2 = posicoes['G']
        return abs(x1 - x2) + abs(y1 - y2)

    problema = ProblemaGrafoComCusto('S', 'G', grafo, posicoes)
    solucao = busca_a_estrela(problema, heuristica_manhattan)
    print("Caminho ótimo:", solucao)  # ['A', 'B', 'G'] - custo 8
