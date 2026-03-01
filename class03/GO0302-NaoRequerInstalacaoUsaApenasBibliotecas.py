# GO0302-NãoRequerInstalaçãoUsaApenasBibliotecas
def busca_generica(problema, fronteira):
    """
    Template genérico para algoritmos de busca
    """
    # Inicialização
    no_inicial = No(problema.estado_inicial)
    fronteira.adicionar(no_inicial)
    explorados = set()  # Estados já visitados

    # Loop principal
    while not fronteira.vazia():
        # 1. Escolher nó da fronteira
        no = fronteira.remover()

        # 2. Testar se é objetivo
        if problema.teste_objetivo(no.estado):
            return no.solucao()  # ✅ Sucesso!

        # 3. Marcar como explorado
        explorados.add(no.estado)

        # 4. Expandir nó (gerar sucessores)
        for acao in problema.acoes(no.estado):
            filho = No(
                estado=problema.resultado(no.estado, acao),
                pai=no,
                acao=acao,
                custo_caminho=no.custo_caminho + problema.custo(no.estado, acao)
            )

            # 5. Adicionar à fronteira se não foi explorado
            if filho.estado not in explorados and not fronteira.contem(filho.estado):
                fronteira.adicionar(filho)

    return None  # ❌ Falha (sem solução)

# Exemplo: Problema simples de grafo
class ProblemaGrafo:
    def __init__(self, inicial, objetivo, grafo):
        self.estado_inicial = inicial
        self.objetivo = objetivo
        self.grafo = grafo  # {estado: [(vizinho, custo), ...]}

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

# Uso com fila (BFS)
from collections import deque
class Fila:
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


if __name__ == "__main__":
    grafo = {'A': [('B', 1), ('C', 2)], 'B': [('D', 3)], 'C': [('D', 1)], 'D': []}
    prob = ProblemaGrafo('A', 'D', grafo)
    solucao = busca_generica(prob, Fila())
    print("Solução:", solucao)  # ['B', 'D'] ou ['C', 'D']
