# GO0301-RepresentaçãoDeCadaPassoDaBusca
class No:
    """
    Representa um nó na árvore de busca
    """
    def __init__(self, estado, pai=None, acao=None, custo_caminho=0):
        self.estado = estado          # Estado do problema
        self.pai = pai                # Nó pai (de onde veio)
        self.acao = acao              # Ação que levou a este estado
        self.custo_caminho = custo_caminho  # Custo acumulado g(n)
        self.profundidade = 0 if pai is None else pai.profundidade + 1

    def __repr__(self):
        return f"No(estado={self.estado}, custo={self.custo_caminho})"

    def caminho(self):
        """Reconstrói caminho da raiz até este nó"""
        no_atual = self
        caminho = []
        while no_atual:
            caminho.append(no_atual.estado)
            no_atual = no_atual.pai
        return list(reversed(caminho))

    def solucao(self):
        """Retorna sequência de ações"""
        no_atual = self
        acoes = []
        while no_atual.pai:
            acoes.append(no_atual.acao)
            no_atual = no_atual.pai
        return list(reversed(acoes))

# Exemplo de uso: construindo uma árvore de busca


if __name__ == "__main__":
    raiz = No(estado="A")
    no_b = No(estado="B", pai=raiz, acao="ir_para_B", custo_caminho=5)
    no_c = No(estado="C", pai=no_b, acao="ir_para_C", custo_caminho=8)

    print(no_c)                    # No(estado=C, custo=8)
    print("Caminho:", no_c.caminho())     # ['A', 'B', 'C']
    print("Ações:", no_c.solucao())       # ['ir_para_B', 'ir_para_C']
    print("Profundidade:", no_c.profundidade)  # 2
