# GO0105-Abordagem1ProgramaçãoTradicionalRegras
def classificar_flor_regras(sepal_length, petal_length):
    if petal_length < 2.5:
        return "setosa"
    elif petal_length < 5.0:
        return "versicolor"
    else:
        return "virginica"
