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
