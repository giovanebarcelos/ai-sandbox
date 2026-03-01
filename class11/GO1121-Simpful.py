# GO1121-Simpful
from simpful import FuzzySystem, LinguisticVariable


if __name__ == "__main__":
    FS = FuzzySystem()
    temp = LinguisticVariable([
        TrapezoidFuzzySet(0, 0, 10, 20, term="cold"),
        TriangleFuzzySet(15, 25, 35, term="warm")
    ])
    FS.add_linguistic_variable("temperature", temp)
    FS.add_rules(["IF temperature IS cold THEN power IS high"])
