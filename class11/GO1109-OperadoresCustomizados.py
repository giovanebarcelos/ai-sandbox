# GO1109-OperadoresCustomizados
# Usar operadores diferentes (padrão é min/max)


if __name__ == "__main__":
    regra_produto = ctrl.Rule(
        antecedent=(qualidade['boa'] & servico['good']),
        consequent=gorjeta['good'],
        and_func=np.multiply  # Usar produto ao invés de min
    )

    # Escolher método de defuzzificação
    gorjeta.defuzzify_method = 'centroid'  # Opções: 'centroid', 'bisector', 'mom', 'som', 'lom'
