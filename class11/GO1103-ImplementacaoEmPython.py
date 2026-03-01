# GO1103-ImplementacaoEmPython
class VariavelLinguistica:
    def __init__(self, nome, universo):
        self.nome = nome
        self.universo = universo
        self.termos = {}

    def adicionar_termo(self, nome, funcao_pertinencia):
        self.termos[nome] = funcao_pertinencia

    def avaliar(self, valor):
        resultado = {}
        for termo, func in self.termos.items():
            resultado[termo] = func(valor)
        return resultado

# Exemplo de uso


if __name__ == "__main__":
    temp = VariavelLinguistica("Temperatura", (0, 50))
    temp.adicionar_termo("FRIO", lambda x: max(0, (15-x)/15))
    temp.adicionar_termo("AGRADÁVEL", 
        lambda x: max(0, min((x-10)/10, (30-x)/10)))
    temp.adicionar_termo("QUENTE", lambda x: max(0, (x-25)/15))

    print(temp.avaliar(18))
    # Saída: {'FRIO': 0.4, 'AGRADÁVEL': 0.6, 'QUENTE': 0.0}
