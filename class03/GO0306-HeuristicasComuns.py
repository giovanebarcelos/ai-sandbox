# GO0306-HeurísticasComuns
class ProblemaLabirinto:
    """
    Modelagem do problema de busca em labirinto
    """
    def __init__(self, labirinto, inicio, objetivo):
        """
        Args:
            labirinto: matriz 2D (0=livre, 1=parede)
            inicio: tupla (linha, coluna)
            objetivo: tupla (linha, coluna)
        """
        self.labirinto = labirinto
        self.linhas = len(labirinto)
        self.colunas = len(labirinto[0])
        self.estado_inicial = inicio
        self.objetivo = objetivo

    def teste_objetivo(self, estado):
        """Testa se chegou no objetivo"""
        return estado == self.objetivo

    def acoes(self, estado):
        """Retorna ações possíveis (movimentos válidos)"""
        linha, coluna = estado
        acoes_possiveis = []

        # Movimentos: Norte, Sul, Leste, Oeste
        movimentos = [
            ('Norte', (-1, 0)),
            ('Sul', (1, 0)),
            ('Leste', (0, 1)),
            ('Oeste', (0, -1))
        ]

        for nome, (dl, dc) in movimentos:
            nova_linha = linha + dl
            nova_coluna = coluna + dc

            # Verifica se está dentro dos limites
            if 0 <= nova_linha < self.linhas and 0 <= nova_coluna < self.colunas:
                # Verifica se não é parede
                if self.labirinto[nova_linha][nova_coluna] == 0:
                    acoes_possiveis.append(nome)

        return acoes_possiveis

    def resultado(self, estado, acao):
        """Aplica ação e retorna novo estado"""
        linha, coluna = estado

        if acao == 'Norte':
            return (linha - 1, coluna)
        elif acao == 'Sul':
            return (linha + 1, coluna)
        elif acao == 'Leste':
            return (linha, coluna + 1)
        elif acao == 'Oeste':
            return (linha, coluna - 1)
        else:
            raise ValueError(f"Ação inválida: {acao}")

    def custo(self, estado, acao):
        """Custo de uma ação (uniforme = 1)"""
        return 1

    def heuristica_manhattan(self, estado):
        """Heurística: distância Manhattan até objetivo"""
        linha, coluna = estado
        obj_linha, obj_coluna = self.objetivo
        return abs(linha - obj_linha) + abs(coluna - obj_coluna)

# Exemplo: Labirinto 5x5 simples


if __name__ == "__main__":
    labirinto = [
        [0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [1, 1, 0, 0, 0]
    ]
    # 0 = livre, 1 = parede
    # Início: (0,0), Objetivo: (4,4)

    problema = ProblemaLabirinto(labirinto, inicio=(0, 0), objetivo=(4, 4))

    # Teste das funcionalidades
    print("Estado inicial:", problema.estado_inicial)
    print("Objetivo:", problema.objetivo)
    print("Ações possíveis de (0,0):", problema.acoes((0, 0)))  # ['Sul', 'Leste']
    print("Resultado de ir ao Sul:", problema.resultado((0, 0), 'Sul'))  # (1, 0)
    print("Heurística de (0,0) até (4,4):", problema.heuristica_manhattan((0, 0)))  # 8
    print("Heurística de (2,2) até (4,4):", problema.heuristica_manhattan((2, 2)))  # 4
