# GO0308-DefinirOProblema
# TAREFA 1: Completar a classe ProblemaLabirinto

class ProblemaLabirinto:
    def __init__(self, labirinto, inicio, objetivo):
        """
        Args:
            labirinto: lista 2D (0=livre, 1=parede)
            inicio: (linha, coluna)
            objetivo: (linha, coluna)
        """
        # TODO: Implementar
        pass

    def teste_objetivo(self, estado):
        """Verifica se é o objetivo"""
        # TODO: Implementar
        pass

    def acoes(self, estado):
        """Retorna ações válidas (Norte, Sul, Leste, Oeste)"""
        # TODO: Implementar movimentos válidos
        # Dica: Verificar limites e paredes
        pass

    def resultado(self, estado, acao):
        """Aplica ação e retorna novo estado"""
        # TODO: Implementar transição de estados
        pass

    def custo(self, estado, acao):
        """Custo de uma ação"""
        return 1  # Uniforme

    def heuristica_manhattan(self, estado):
        """Heurística Manhattan"""
        # TODO: Calcular |x1-x2| + |y1-y2|
        pass

# Labirinto de teste (10x10)


if __name__ == "__main__":
    lab = [
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 1, 0, 1, 0]
    ]

    problema = ProblemaLabirinto(lab, (0, 0), (9, 9))
