# GO1109-OperadoresCustomizados
# Usar operadores diferentes (padrão é min/max)

# CONTEXTO: este snippet complementa GO1108. Requer que as variáveis
# 'qualidade', 'servico' e 'gorjeta' (ctrl.Antecedent/Consequent) já
# estejam definidas na sessão (execute GO1108 antes).
#
# DEMONSTRA dois pontos de customização do scikit-fuzzy:
#   1. Trocar o operador AND padrão (min) pelo produto algébrico
#   2. Escolher o método de defuzzificação
#
# Para outro problema: substitua os termos das variáveis e escolha o
# método de defuzzificação mais adequado ao seu domínio.

if __name__ == "__main__":
    regra_produto = ctrl.Rule(
        antecedent=(qualidade['boa'] & servico['good']),
        consequent=gorjeta['good'],
        and_func=np.multiply  # Usar produto ao invés de min
    )

    # Escolher método de defuzzificação
    # Opções: 'centroid' (mais preciso), 'bisector', 'mom', 'som', 'lom'.
    # Veja GO1106 para entender a diferença entre eles.
    gorjeta.defuzzify_method = 'centroid'  # Opções: 'centroid', 'bisector', 'mom', 'som', 'lom'
