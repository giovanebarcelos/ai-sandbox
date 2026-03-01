# GO0304-NãoRequerInstalaçãoUsaApenasBibliotecas
class PilhaLIFO:
    """Pilha para DFS"""
    def __init__(self):
        self.pilha = []

    def adicionar(self, no):
        self.pilha.append(no)

    def remover(self):
        return self.pilha.pop()  # Remove do FINAL (LIFO)

    def vazia(self):
        return len(self.pilha) == 0

    def contem(self, estado):
        return any(no.estado == estado for no in self.pilha)

def busca_profundidade(problema, limite=None):
    """
    Busca em Profundidade (DFS)

    Args:
        problema: Instância do problema
        limite: Profundidade máxima (None = sem limite)

    Retorna: lista de ações ou None
    """
    # Nó inicial
    no = No(problema.estado_inicial)

    # Teste trivial
    if problema.teste_objetivo(no.estado):
        return no.solucao()

    # Fronteira: PILHA
    fronteira = PilhaLIFO()
    fronteira.adicionar(no)

    # Estados explorados
    explorados = set()
    nos_explorados = 0

    while not fronteira.vazia():
        # Remove o ÚLTIMO da pilha (LIFO)
        no = fronteira.remover()
        nos_explorados += 1

        # Adiciona aos explorados
        explorados.add(no.estado)

        # Verifica limite de profundidade
        if limite and no.profundidade >= limite:
            continue  # Poda este ramo

        # Expande nó
        for acao in problema.acoes(no.estado):
            filho = No(
                estado=problema.resultado(no.estado, acao),
                pai=no,
                acao=acao,
                custo_caminho=no.custo_caminho + 1
            )

            # Verifica se já foi explorado
            if filho.estado not in explorados and not fronteira.contem(filho.estado):
                # Testa objetivo
                if problema.teste_objetivo(filho.estado):
                    print(f"✅ Solução encontrada após explorar {nos_explorados} nós")
                    return filho.solucao()

                # Adiciona à fronteira
                fronteira.adicionar(filho)

    print(f"❌ Sem solução (explorou {nos_explorados} nós)")
    return None

# Exemplo: Comparar DFS com e sem limite de profundidade
class ProblemaArvore:
    def __init__(self, inicial, objetivo, arvore):
        self.estado_inicial = inicial
        self.objetivo = objetivo
        self.arvore = arvore  # {estado: [filhos]}

    def teste_objetivo(self, estado):
        return estado == self.objetivo

    def acoes(self, estado):
        # Retorna filhos na ordem (DFS explora da direita para esquerda em LIFO)
        return self.arvore.get(estado, [])

    def resultado(self, estado, acao):
        return acao

# Árvore mais profunda: A → B → D → H (G está aqui no nível 3)
#                       A → C → F (ramo alternativo)
# B é explorado primeiro (LIFO: último adicionado = primeiro removido)


if __name__ == "__main__":
    arvore = {
        'A': ['C', 'B'],  # B adicionado por último, removido primeiro (LIFO)
        'B': ['E', 'D'],  # D adicionado por último, explorado primeiro
        'D': ['I', 'H'],  # H adicionado por último
        'H': ['G'],       # G está profundo (nível 4)
        'C': ['F'],
        'E': [], 'F': [], 'I': [], 'G': []
    }
    problema = ProblemaArvore('A', 'G', arvore)

    print("DFS sem limite (explora fundo):")
    solucao1 = busca_profundidade(problema)
    print("Caminho:", solucao1)  # ['B', 'D', 'H', 'G'] - explora profundamente

    print("\nDFS com limite=2 (para no nível 2):")
    solucao2 = busca_profundidade(problema, limite=2)
    print("Caminho:", solucao2 if solucao2 else "Não encontrou (G está no nível 4)")
