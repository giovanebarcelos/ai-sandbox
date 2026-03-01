# GO1102-ImplementacaoEmPython
# Operadores fuzzy
def fuzzy_and_min(a, b):
    return min(a, b)

def fuzzy_and_prod(a, b):
    return a * b

def fuzzy_or_max(a, b):
    return max(a, b)

def fuzzy_not(a):
    return 1 - a

# Exemplo de uso


if __name__ == "__main__":
    temp_quente = 0.8
    umid_alta = 0.6

    # Regra: SE quente AND alta THEN ligar AC
    forca = fuzzy_and_min(temp_quente, umid_alta)
    print(f"Ligar AC: {forca}")
