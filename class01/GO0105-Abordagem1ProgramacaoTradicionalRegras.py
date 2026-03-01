# GO0105-Abordagem1ProgramaçãoTradicionalRegras
def classificar_flor_regras(sepal_length, petal_length):
    if petal_length < 2.5:
        return "setosa"
    elif petal_length < 5.0:
        return "versicolor"
    else:
        return "virginica"


if __name__ == '__main__':
    # Testa classificação de espécies de íris por regras
    exemplos = [
        (5.1, 1.4, "setosa"),
        (6.0, 4.5, "versicolor"),
        (7.0, 6.0, "virginica"),
    ]
    print("=== Classificador por Regras ===")
    for sepal, petal, esperado in exemplos:
        resultado = classificar_flor_regras(sepal, petal)
        status = "✅" if resultado == esperado else "❌"
        print(f"  sepal={sepal}, petal={petal} → {resultado} {status}")
