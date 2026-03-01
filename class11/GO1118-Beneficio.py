# GO1118-Beneficio
import skfuzzy as fuzz
from sklearn.ensemble import RandomForestClassifier

# 1. Fuzzificar features
def fuzzify_age(age):
    young = fuzz.trapmf(age, [0, 0, 20, 35])
    adult = fuzz.trimf(age, [25, 40, 60])
    senior = fuzz.trapmf(age, [50, 70, 100, 100])
    return [young, adult, senior]


if __name__ == "__main__":
    X_fuzzy = np.array([fuzzify_age(age) for age in X[:, 0]])

    # 2. Treinar ML com features fuzzy
    rf = RandomForestClassifier()
    rf.fit(X_fuzzy, y)
