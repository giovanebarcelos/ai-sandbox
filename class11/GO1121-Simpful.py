# GO1121-Simpful
# Simpful — biblioteca alternativa ao scikit-fuzzy com sintaxe mais próxima
# da linguagem natural. Vantagem: regras escritas como strings legíveis
# ("IF temperature IS cold THEN power IS high").
#
# Para usar: pip install simpful
# Para outro problema: substitua os termos (cold/warm) e variáveis
# (temperature/power) pelos do seu domínio. As classes TrapezoidFuzzySet
# e TriangleFuzzySet recebem os mesmos parâmetros (a,b,c,d) / (a,b,c)
# usados nos outros exemplos desta aula.
from simpful import FuzzySystem, LinguisticVariable


if __name__ == "__main__":
    FS = FuzzySystem()
    temp = LinguisticVariable([
        TrapezoidFuzzySet(0, 0, 10, 20, term="cold"),
        TriangleFuzzySet(15, 25, 35, term="warm")
    ])
    FS.add_linguistic_variable("temperature", temp)
    FS.add_rules(["IF temperature IS cold THEN power IS high"])
