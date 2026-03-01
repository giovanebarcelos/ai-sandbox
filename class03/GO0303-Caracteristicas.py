# GO0303-Características
from collections import deque

class FilaFIFO:
    """Fila para BFS"""
    def __init__(self):
        self.fila = deque()

    def adicionar(self, no):
        self.fila.append(no)

    def remover(self):
        return self.fila.popleft()

    def vazia(self):
        return len(self.fila) == 0

    def contem(self, estado):
        return any(no.estado == estado for no in self.fila)

def busca_largura(problema):
    """
    Busca em Largura (BFS)
    Retorna: lista de ações ou None
    """
    # Nó inicial
    no = No(problema.estado_inicial)

    # Teste trivial
    if problema.teste_objetivo(no.estado):
        return no.solucao()

    # Fronteira: FILA
    fronteira = FilaFIFO()
    fronteira.adicionar(no)

    # Estados explorados
    explorados = set()
    nos_explorados = 0

    while not fronteira.vazia():
        # Remove o PRIMEIRO da fila (FIFO)
        no = fronteira.remover()
        nos_explorados += 1

        # Adiciona aos explorados
        explorados.add(no.estado)

        # Expande nó
        for acao in problema.acoes(no.estado):
            filho = No(
                estado=problema.resultado(no.estado, acao),
                pai=no,
                acao=acao,
                custo_caminho=no.custo_caminho + 1
            )

            # Verifica se já foi explorado ou está na fronteira
            if filho.estado not in explorados and not fronteira.contem(filho.estado):
                # Testa objetivo
                if problema.teste_objetivo(filho.estado):
                    print(f"✅ Solução encontrada após explorar {nos_explorados} nós")
                    return filho.solucao()

                # Adiciona à fronteira
                fronteira.adicionar(filho)

    print(f"❌ Sem solução (explorou {nos_explorados} nós)")
    return None

# Exemplo: Buscar caminho em grafo simples
class ProblemaGrafoSimples:
    def __init__(self, inicial, objetivo, grafo):
        self.estado_inicial = inicial
        self.objetivo = objetivo
        self.grafo = grafo  # {estado: [vizinhos]}

    def teste_objetivo(self, estado):
        return estado == self.objetivo

    def acoes(self, estado):
        return self.grafo.get(estado, [])

    def resultado(self, estado, acao):
        return acao

# Grafo: A → B → D, A → C → D


if __name__ == "__main__":
    grafo = {'A': ['B', 'C'], 'B': ['D', 'E'], 'C': ['D', 'F'], 'D': [], 'E': [], 'F': []}
    problema = ProblemaGrafoSimples('A', 'D', grafo)
    solucao = busca_largura(problema)
    print("Caminho encontrado:", solucao)  # ['B', 'D'] (primeiro caminho em largura)
