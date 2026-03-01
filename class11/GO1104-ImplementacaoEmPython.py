# GO1104-ImplementacaoEmPython
class RegraFuzzy:
    def __init__(self, antecedentes, consequente):
        """
        antecedentes: lista de tuplas (variavel, termo, valor)
        consequente: tupla (variavel, termo)
        """
        self.antecedentes = antecedentes
        self.consequente = consequente

    def avaliar(self):
        # Calcular força da regra (AND = min)
        forcas = [valor for _, _, valor in self.antecedentes]
        return min(forcas)

# Exemplo de uso


if __name__ == "__main__":
    regra1 = RegraFuzzy(
        antecedentes=[
            ("temperatura", "QUENTE", 0.7),
            ("umidade", "ALTA", 0.6)
        ],
        consequente=("potencia_ac", "MÁXIMO")
    )

    forca = regra1.avaliar()
    print(f"Regra ativada com força: {forca}")
