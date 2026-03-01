# GO1113-ImplementacaoConceitual
# pip install type2fuzzy
from type2fuzzy import Type2FuzzySet

# Criar conjunto Type-2


if __name__ == "__main__":
    temp_quente = Type2FuzzySet(
        lower_mf=triangular([20, 25, 30]),
        upper_mf=triangular([18, 25, 32])
    )
