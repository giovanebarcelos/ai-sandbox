# GO1118-Beneficio
import skfuzzy as fuzz
from sklearn.ensemble import RandomForestClassifier

# BLOCO 1 — FUZZIFICAÇÃO: transforma 'age' numérico em 3 graus linguísticos.
# Para outro problema: crie uma função fuzzify_X para cada feature que
# se beneficia de representação fuzzy (ex: renda, pressão, temperatura).
def fuzzify_age(age):
    young = fuzz.trapmf(age, [0, 0, 20, 35])
    adult = fuzz.trimf(age, [25, 40, 60])
    senior = fuzz.trapmf(age, [50, 70, 100, 100])
    return [young, adult, senior]


if __name__ == "__main__":
    # BLOCO 2 — TRANSFORMAÇÃO: aplica fuzzificação em todo o dataset.
    # Para outro problema: combine fuzzify de múltiplas features e concatene
    # as colunas (ex: np.hstack([X_fuzzy_age, X_fuzzy_income])).
    X_fuzzy = np.array([fuzzify_age(age) for age in X[:, 0]])

    # BLOCO 3 — MODELO ML: qualquer classificador/regressor pode ser usado aqui.
    # A vantagem das features fuzzy é que o modelo recebe contexto linguístico
    # em vez de apenas valores numéricos crus.
    rf = RandomForestClassifier()
    rf.fit(X_fuzzy, y)
