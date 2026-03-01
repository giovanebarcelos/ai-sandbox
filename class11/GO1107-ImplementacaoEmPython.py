# GO1107-ImplementacaoEmPython
def sugeno_inference_order0(weights, outputs):
    """
    Sugeno ordem 0 (consequentes constantes)
    weights: lista de forças das regras
    outputs: lista de valores constantes
    """
    numerator = sum(w * z for w, z in zip(weights, outputs))
    denominator = sum(weights)
    return numerator / denominator if denominator > 0 else 0

def sugeno_inference_order1(weights, coeffs, inputs):
    """
    Sugeno ordem 1 (consequentes lineares)
    weights: forças das regras
    coeffs: lista de coeficientes [p, q, r] por regra
    inputs: valores de entrada [x, y]
    """
    outputs = []
    for c in coeffs:
        z = c[0] * inputs[0] + c[1] * inputs[1] + c[2]
        outputs.append(z)
    return sugeno_inference_order0(weights, outputs)

# Exemplo de uso


if __name__ == "__main__":
    weights = [0.0, 0.5, 0.5]
    outputs = [5, 15, 25]
    gorjeta = sugeno_inference_order0(weights, outputs)
    print(f"Gorjeta Sugeno: {gorjeta}%")  # Saída: 20%
