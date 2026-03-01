# GO1105-ImplementacaoPythonSimplificada
def mamdani_inference(inputs, rules, output_mfs):
    # 1. Fuzzificação (já feita em inputs)

    # 2. Inferência
    output_fuzzy = {}
    for rule in rules:
        strength = rule.evaluate(inputs)
        output_term = rule.consequent
        if output_term not in output_fuzzy:
            output_fuzzy[output_term] = 0
        output_fuzzy[output_term] = max(
            output_fuzzy[output_term], strength)

    # 3. Defuzzificação (centroide)
    numerator = 0
    denominator = 0
    for x in range(0, 26):
        mu = 0
        for term, strength in output_fuzzy.items():
            mu = max(mu, min(strength, output_mfs[term](x)))
        numerator += mu * x
        denominator += mu

    return numerator / denominator if denominator > 0 else 0


if __name__ == '__main__':
    print("=== Demonstração de Inferência Mamdani ===")

    # Funções de pertinência triangular inline
    def tri(x, a, b, c):
        if x <= a or x >= c:
            return 0.0
        elif x <= b:
            return (x - a) / (b - a)
        else:
            return (c - x) / (c - b)

    # Regra simples: se entrada for "media" → saída é "moderada"
    class SimpleRule:
        def __init__(self, strength, consequent):
            self._strength = strength
            self.consequent = consequent
        def evaluate(self, inputs):
            return self._strength

    rules = [
        SimpleRule(0.8, 'baixo'),
        SimpleRule(0.4, 'alto'),
    ]

    output_mfs = {
        'baixo': lambda x: tri(x,  0,  6, 12),
        'alto':  lambda x: tri(x, 14, 20, 25),
    }

    resultado = mamdani_inference({}, rules, output_mfs)
    print(f"  Saída defuzzificada (Centróide): {resultado:.2f}")
    print("  (Esperado: valor próximo de 6–10, dominado pela regra 'baixo' mais forte)")
